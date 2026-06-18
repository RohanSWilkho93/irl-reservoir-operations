"""
airl/agent.py
=============
The AIRL agent: a BC-warm-started actor (the shared iqlearn ParametricPolicy) +
a value critic + an AIRL discriminator (reward g(s,a) + shaping h(s)).

Training:
  * discriminator — BCE(expert=1, policy=0) + gradient penalty + label smoothing.
  * policy/critic  — PPO (clipped surrogate, GAE) on the discriminator-logit
    reward, with an entropy bonus and a KL-to-BC anchor (the family's closed-form
    KL).  The actor's family (Beta/LogNormal/Hard/SoftGating) is inherited from
    the BC checkpoint, so the same code trains every family.

Inference (`select_action`) matches iqlearn so iqlearn.environment.ReservoirRollout
can roll the agent out for evaluation and the Monte-Carlo result fans.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import asdict
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from iqlearn.networks.policy import build_policy_network
from iqlearn.distributions import make_distribution
from airl.config import AIRLConfig
from airl.networks import CriticNetwork, AIRLDiscriminator
from airl.environment import ReplayBuffer, RolloutBuffer

POLICY_TYPE = "airl"


class AIRLAgent:
    def __init__(self, config: AIRLConfig, bc_ckpt: dict, device):
        self.config = config
        self.device = torch.device(device)
        self.bc_config_dict = bc_ckpt["config"]
        bc_cfg = SimpleNamespace(**self.bc_config_dict)
        self.policy_family = bc_cfg.policy_family
        self.dist_params = getattr(bc_cfg, "dist_params", {}) or {}
        self.distribution = make_distribution(self.policy_family, self.dist_params)

        # actor (warm-started) + frozen BC prior for the KL anchor
        self.policy = build_policy_network(bc_cfg).to(self.device)
        self.policy.load_state_dict(bc_ckpt["state_dict"])
        self.bc_policy = build_policy_network(bc_cfg).to(self.device)
        self.bc_policy.load_state_dict(bc_ckpt["state_dict"])
        self.bc_policy.eval()
        for p in self.bc_policy.parameters():
            p.requires_grad_(False)

        self.critic = CriticNetwork(config.state_dim, config.critic_hidden_dim,
                                    config.critic_n_hidden_layers).to(self.device)
        self.discriminator = AIRLDiscriminator(
            config.state_dim, config.action_dim, self.policy, self.distribution,
            config.disc_hidden_dim, config.disc_n_hidden_layers, config.gamma).to(self.device)

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=config.lr_policy)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        # discriminator optimiser: ONLY the reward + shaping nets (policy is a
        # reference inside the discriminator and must not be stepped here).
        disc_params = list(self.discriminator.reward_net.parameters()) + \
                      list(self.discriminator.shaping_net.parameters())
        self.disc_opt = optim.Adam(disc_params, lr=config.lr_discriminator)
        self._disc_params = disc_params

        self.expert_buffer = ReplayBuffer(config.expert_buffer_size)
        self.policy_buffer = ReplayBuffer(config.policy_buffer_size)
        self.training_stats = defaultdict(list)
        self._last_disc = {"expert_acc": 0.5, "policy_acc": 0.5}

    # -----------------------------------------------------------------------
    # Inference (iqlearn-compatible)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, states, deterministic=True, generator=None):
        params = self.policy(states)
        if deterministic:
            return self.distribution.mean_action(params)
        return self.distribution.rsample(params, generator=generator)[0]

    # -----------------------------------------------------------------------
    # Expert data
    # -----------------------------------------------------------------------

    def add_expert_data(self, states, actions, next_states, dones):
        for s, a, ns, d in zip(states, actions, next_states, dones):
            self.expert_buffer.push(s, a, ns, d)

    # -----------------------------------------------------------------------
    # Discriminator update
    # -----------------------------------------------------------------------

    def _gradient_penalty(self, e_batch, p_batch):
        to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        e_s, e_a, e_ns, _ = [to_t(x) for x in e_batch]
        p_s, p_a, p_ns, _ = [to_t(x) for x in p_batch]
        alpha = torch.rand(e_s.shape[0], 1, device=self.device)
        i_s = (alpha * e_s + (1 - alpha) * p_s).requires_grad_(True)
        i_a = (alpha * e_a + (1 - alpha) * p_a).requires_grad_(True)
        i_ns = (alpha * e_ns + (1 - alpha) * p_ns).requires_grad_(True)
        f = self.discriminator.compute_f(i_s, i_a, i_ns)
        grads = torch.autograd.grad(f, [i_s, i_a, i_ns], torch.ones_like(f),
                                    create_graph=True, retain_graph=True)
        return ((torch.cat([g.reshape(e_s.shape[0], -1) for g in grads], 1).norm(2, dim=1) - 1) ** 2).mean()

    def update_discriminator(self, batch_size, num_updates):
        half = batch_size // 2
        if len(self.expert_buffer) < half or len(self.policy_buffer) < half:
            return {"disc_loss": 0.0, "expert_acc": 0.5, "policy_acc": 0.5}
        eps = self.config.label_smoothing_epsilon
        to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        tot_loss = tot_e = tot_p = 0.0
        for _ in range(num_updates):
            e_batch = self.expert_buffer.sample(half)
            p_batch = self.policy_buffer.sample(half)
            e_s, e_a, e_ns, _ = [to_t(x) for x in e_batch]
            p_s, p_a, p_ns, _ = [to_t(x) for x in p_batch]
            e_out, p_out = self.discriminator(e_s, e_a, e_ns), self.discriminator(p_s, p_a, p_ns)
            loss = (F.binary_cross_entropy(e_out, torch.ones_like(e_out) * (1 - eps))
                    + F.binary_cross_entropy(p_out, torch.zeros_like(p_out) + eps))
            gp = self._gradient_penalty(e_batch, p_batch)
            self.disc_opt.zero_grad()
            (loss + self.config.gradient_penalty_coef * gp).backward()
            torch.nn.utils.clip_grad_norm_(self._disc_params, self.config.max_grad_norm)
            self.disc_opt.step()
            with torch.no_grad():
                tot_loss += loss.item()
                tot_e += (self.discriminator(e_s, e_a, e_ns) > 0.5).float().mean().item()
                tot_p += (self.discriminator(p_s, p_a, p_ns) < 0.5).float().mean().item()
        stats = {"disc_loss": tot_loss / num_updates, "expert_acc": tot_e / num_updates,
                 "policy_acc": tot_p / num_updates}
        self._last_disc = stats
        for k, v in stats.items():
            self.training_stats[k].append(v)
        return stats

    # -----------------------------------------------------------------------
    # PPO update
    # -----------------------------------------------------------------------

    def _gae(self, rewards, values, next_values, dones):
        gamma, lam = self.config.gamma, self.config.gae_lambda
        adv = torch.zeros_like(rewards)
        last = 0.0
        for t in reversed(range(len(rewards))):
            nv = 0.0 if dones[t] > 0.5 else (next_values[t] if t == len(rewards) - 1 else values[t + 1])
            delta = rewards[t] + gamma * nv - values[t]
            last = delta if dones[t] > 0.5 else delta + gamma * lam * last
            adv[t] = last
        return adv, adv + values

    def collect_rollout(self, env, min_steps, deterministic=False):
        rb = RolloutBuffer()
        steps = 0
        while steps < min_steps:
            state, done = env.reset(), False
            while not done:
                st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    params = self.policy(st)
                    if deterministic:
                        action = self.distribution.mean_action(params)
                        logp = self.distribution.log_prob(params, action)
                    else:
                        action, logp = self.distribution.rsample(params)
                    value = self.critic(st).squeeze(-1)
                a = float(action.item())
                ns, _, done, _ = env.step(a)
                with torch.no_grad():
                    ns_t = torch.as_tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(0)
                    a_t = torch.as_tensor([[a]], dtype=torch.float32, device=self.device)
                    reward = self.discriminator.get_reward(st, a_t, ns_t).item()
                rb.push(state, [a], reward, ns, float(done), float(logp.item()), float(value.item()))
                self.policy_buffer.push(state, [a], ns, done)
                state = ns
                steps += 1
        return rb.get(self.device)

    def update_policy_ppo(self, rollout):
        states, actions = rollout["states"], rollout["actions"]
        old_lp, rewards = rollout["log_probs"], rollout["rewards"]
        next_states, dones, old_values = rollout["next_states"], rollout["dones"], rollout["values"]
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1)
        adv, returns = self._gae(rewards, old_values, next_values, dones)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        with torch.no_grad():
            bc_params = self.bc_policy(states)
        tot_p = tot_c = tot_kl = 0.0
        for _ in range(self.config.ppo_epochs):
            params = self.policy(states)
            new_lp = self.distribution.log_prob(params, actions)
            entropy = self.distribution.entropy(params).mean()
            new_v = self.critic(states).squeeze(-1)
            ratio = torch.exp(new_lp - old_lp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * adv
            p_loss = -torch.min(s1, s2).mean()
            kl = (self.config.kl_regularization_coef * self.distribution.kl(params, bc_params).mean()
                  if self.config.kl_regularization_coef > 0 else torch.zeros((), device=self.device))
            loss = p_loss + kl - self.config.entropy_coef * entropy
            c_loss = F.mse_loss(new_v, returns)
            self.policy_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.policy_opt.step()
            self.critic_opt.zero_grad(); c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_opt.step()
            tot_p += p_loss.item(); tot_c += c_loss.item(); tot_kl += float(kl.item())
        n = self.config.ppo_epochs
        stats = {"policy_loss": tot_p / n, "critic_loss": tot_c / n, "kl_loss": tot_kl / n,
                 "mean_reward": rewards.mean().item()}
        for k, v in stats.items():
            self.training_stats[k].append(v)
        return stats

    # -----------------------------------------------------------------------
    # Adversarial schedule
    # -----------------------------------------------------------------------

    def warmup_discriminator(self, env, iterations):
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.policy.eval()
        for _ in range(iterations):
            self.collect_rollout(env, self.config.steps_per_iteration, deterministic=True)
            self.update_discriminator(self.config.batch_size, self.config.warmup_disc_updates)
        self.policy.train()
        for p in self.policy.parameters():
            p.requires_grad_(True)

    def train_with_validation(self, train_env, val_rollout, trial=None):
        from airl.scoring import composite_score, rollout_fidelity
        best_score, best_state = -float("inf"), None
        stall = 0
        for it in range(self.config.num_iterations):
            if it % 20 == 0 and it > 0:
                self.policy_buffer.clear()
            rollout = self.collect_rollout(train_env, self.config.steps_per_iteration)
            self.update_discriminator(self.config.batch_size, self.config.disc_updates)
            self.update_policy_ppo(rollout)
            if it % self.config.eval_interval == 0:
                self.policy.eval()
                traj = val_rollout.rollout(self, deterministic=True)
                self.policy.train()
                fid = rollout_fidelity(traj)
                score = composite_score(fid["release_corr"], fid["storage_corr"],
                                        fid["release_nrmse"], fid["storage_nrmse"],
                                        self._last_disc["expert_acc"], self._last_disc["policy_acc"])
                self.training_stats["val_score"].append(score)
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}
                    stall = 0
                else:
                    stall += 1
                if trial is not None:
                    trial.report(score, it)
                    if trial.should_prune():
                        import optuna
                        raise optuna.TrialPruned()
                if stall >= max(1, self.config.early_stopping_patience // self.config.eval_interval):
                    break
        if best_state is not None:
            self.policy.load_state_dict(best_state)
        return best_score

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path):
        torch.save({
            "policy": {k: v.detach().cpu() for k, v in self.policy.state_dict().items()},
            "critic": {k: v.detach().cpu() for k, v in self.critic.state_dict().items()},
            "reward_net": {k: v.detach().cpu() for k, v in self.discriminator.reward_net.state_dict().items()},
            "shaping_net": {k: v.detach().cpu() for k, v in self.discriminator.shaping_net.state_dict().items()},
            "airl_config": asdict(self.config),
            "bc_config": self.bc_config_dict,
            "policy_type": POLICY_TYPE,
            "policy_family": self.policy_family,
        }, path)

    @classmethod
    def from_checkpoint(cls, path, device):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = AIRLConfig(**ckpt["airl_config"]); cfg.device = str(device)
        bc_ckpt = {"state_dict": ckpt["policy"], "config": ckpt["bc_config"], "policy_type": POLICY_TYPE}
        agent = cls(cfg, bc_ckpt, device)
        agent.policy.load_state_dict(ckpt["policy"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.discriminator.reward_net.load_state_dict(ckpt["reward_net"])
        agent.discriminator.shaping_net.load_state_dict(ckpt["shaping_net"])
        agent.policy.eval()
        return agent
