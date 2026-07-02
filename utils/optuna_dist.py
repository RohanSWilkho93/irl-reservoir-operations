"""
utils/optuna_dist.py
====================
Shared Optuna helpers for **local, file-based distributed tuning** — no internet,
no database server.  Scheduler-agnostic: works with plain background processes,
`srun`/`sbatch`, PBS, or anything that can launch processes on a shared filesystem.

Two modes, chosen by whether `storage` (a local file path) is given:

  * storage is None  -> a plain in-memory study; `n_jobs` threads inside ONE
    process (the default, single-machine behaviour).
  * storage is a path -> a shared JournalFileStorage study on a local (shared)
    filesystem.  Launch multiple worker processes, all pointing at the same
    `storage` file + `study_name`; Optuna hands each worker distinct trials.
    Resumable, multi-node, no network.

`run_optimize` caps the **global** completed-trial count at `n_trials` via
MaxTrialsCallback, so any number of workers together stop at `n_trials` total
(otherwise each worker would run its own `n_trials`).

Roles (for the multi-process workflow):
  worker   -> contribute trials to the shared study, then exit (no retrain/save).
  finalize -> attach to the completed study and do retrain+save (no new trials).
  full     -> single process: optimize then retrain+save (the default when no
              --storage; also handy for a single resumable shared run).
"""

from __future__ import annotations

import os
from pathlib import Path

import optuna
from optuna.trial import TrialState


def _journal_backend(path: str):
    """
    Version- and platform-tolerant JournalFileStorage backend.

    Locking: on Linux/HPC (incl. NFS/Lustre) the default symlink lock is the
    robust choice.  On Windows, symlinks need elevated privilege, so use the
    open-file lock instead — this only matters for local dev; the cluster path
    keeps the symlink lock.
    """
    st = optuna.storages
    if hasattr(st, "journal") and hasattr(st.journal, "JournalFileBackend"):
        j = st.journal                                       # Optuna >= 4.0
        if os.name == "nt" and hasattr(j, "JournalFileOpenLock"):
            return j.JournalFileBackend(str(path), lock_obj=j.JournalFileOpenLock(str(path)))
        return j.JournalFileBackend(str(path))               # symlink lock (Linux/NFS default)
    return st.JournalFileStorage(str(path))                  # older Optuna


def build_study(*, direction: str, sampler, pruner=None,
                storage: str | None = None, study_name: str | None = None):
    """
    Create (or attach to) a study.  Returns (study, is_shared).

    storage=None  -> in-memory.
    storage=path  -> JournalFileStorage on that local path; `load_if_exists=True`
                     so every worker + the finalize step share ONE study.
    """
    if storage is None:
        return optuna.create_study(direction=direction, sampler=sampler, pruner=pruner), False
    p = Path(storage)
    p.parent.mkdir(parents=True, exist_ok=True)
    backend = optuna.storages.JournalStorage(_journal_backend(p))
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner,
                                storage=backend, study_name=study_name, load_if_exists=True)
    return study, True


def load_study(*, storage: str, study_name: str):
    """Attach to an existing shared study (used by the `finalize` role)."""
    backend = optuna.storages.JournalStorage(_journal_backend(storage))
    return optuna.load_study(study_name=study_name, storage=backend)


def run_optimize(study, objective, *, n_trials: int, n_jobs: int, shared: bool):
    """
    Run trials.  When `shared`, stop when the study's GLOBAL completed count
    reaches `n_trials` (so N workers together do n_trials, not N*n_trials).

    Note: the cap is approximate — a worker can finish up to (n_jobs - 1) extra
    in-flight trials before it stops.  With one trial per worker process (n_jobs=1),
    the total overshoot is at most (#workers - 1) trials.
    """
    if shared:
        cb = optuna.study.MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[cb])
    else:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)


def n_completed(study) -> int:
    """Number of COMPLETE trials (for logging / finalize sanity checks)."""
    return sum(t.state == TrialState.COMPLETE for t in study.get_trials(deepcopy=False))
