"""
utils/runs.py
=============
Shared, stage-agnostic helpers used by every algorithm script
(bc_tuning.py, iq_tuning.py, generate_results.py):

  * device resolution        — _resolve_device
  * comment-preserving YAML  — _deep_update, _writeback_yaml
  * run-folder allocation    — _resolve_run_id   (write side, used by tune.py)
  * run-folder lookup        — _find_run_folder  (read side, used downstream)

Folder naming convention
------------------------
    results/<reservoir>/iqlearn/<run_id>/

`run_id` is a positive integer, auto-incremented by tune.py or supplied
explicitly via --run_id.  A SINGLE run folder holds the artifacts of every
stage for that run:

    <run_id>/
        bc_best_config.json   bc_policy.pt
        iq_best_config.json   iq_agent.pt
        run_args.json         ...

so downstream stages locate it by run_id alone.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import torch
import yaml


# Run folders are bare positive integers: "1", "2", "3", ...
_RUN_FOLDER_PATTERN = re.compile(r"^(\d+)$")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the actual available device string."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# ---------------------------------------------------------------------------
# YAML write-back (comment-preserving)
# ---------------------------------------------------------------------------

def _deep_update(target: dict, source: dict) -> None:
    """
    Recursively update target with values from source.

    Descends into nested dicts instead of replacing them wholesale.
    Compatible with ruamel.yaml CommentedMap so YAML comments survive
    the write-back.
    """
    for key, val in source.items():
        if key in target and hasattr(target[key], "items") and hasattr(val, "items"):
            _deep_update(target[key], val)
        else:
            target[key] = val


def _writeback_yaml(path: Path, updates: dict) -> None:
    """
    Merge updates into an existing YAML file and write it back.

    Uses ruamel.yaml (comment-preserving) if installed; falls back to plain
    PyYAML with a warning.
    """
    try:
        from ruamel.yaml import YAML
        ryaml = YAML()
        ryaml.preserve_quotes = True
        ryaml.best_width = 4096
        with open(path, "r") as f:
            doc = ryaml.load(f)
        _deep_update(doc, updates)
        with open(path, "w") as f:
            ryaml.dump(doc, f)
    except ImportError:
        warnings.warn(
            "ruamel.yaml not installed — writing YAML with plain PyYAML. "
            "YAML comments will be lost. Install ruamel.yaml to preserve them.",
            UserWarning,
            stacklevel=2,
        )
        with open(path, "r") as f:
            doc = yaml.safe_load(f)
        _deep_update(doc, updates)
        with open(path, "w") as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Run-folder allocation  (write side — used by tune.py)
# ---------------------------------------------------------------------------

def _resolve_run_id(
    base_dir:   Path,
    run_id_arg: int | None,
) -> tuple[int, Path]:
    """
    Resolve the integer run_id and return the corresponding run folder Path.

    If run_id_arg is None, scans base_dir for existing '<int>' folders and
    returns highest + 1 (or 1 if none).  If provided, uses it directly and
    warns if the folder already exists (contents will be overwritten).

    Returns
    -------
    (run_id, run_folder)
    """
    existing_ids: list[int] = []
    if base_dir.exists():
        for d in base_dir.iterdir():
            m = _RUN_FOLDER_PATTERN.match(d.name)
            if d.is_dir() and m:
                existing_ids.append(int(m.group(1)))

    run_id = (max(existing_ids, default=0) + 1) if run_id_arg is None else run_id_arg
    run_folder = base_dir / f"{run_id}"

    if run_folder.exists() and run_id_arg is not None:
        print(
            f"\nWARNING: Run folder already exists: {run_folder}\n"
            f"  Contents will be overwritten.\n",
            file=sys.stderr,
        )

    return run_id, run_folder


# ---------------------------------------------------------------------------
# Run-folder lookup  (read side — used by iq_tuning.py / generate_results.py)
# ---------------------------------------------------------------------------

def _find_run_folder(base_dir: Path, run_id: int) -> Path:
    """
    Locate the existing run folder named '<run_id>' under base_dir.

    Parameters
    ----------
    base_dir : Path to results/<reservoir>/iqlearn/
    run_id   : Integer run identifier supplied via --run_id.

    Returns
    -------
    Path  The matched run folder.

    Exits with a clear error if base_dir does not exist or no folder matches.
    """
    if not base_dir.exists():
        sys.exit(
            f"\nERROR: Results directory does not exist: {base_dir}\n"
            f"  Run tune.py for this reservoir first.\n"
        )

    run_folder = base_dir / f"{run_id}"
    if run_folder.is_dir():
        return run_folder

    available = sorted(
        (d.name for d in base_dir.iterdir()
         if d.is_dir() and _RUN_FOLDER_PATTERN.match(d.name)),
        key=int,
    )
    avail_str = ", ".join(available) if available else "none found"
    sys.exit(
        f"\nERROR: No run folder found for run_id={run_id} in:\n"
        f"  {base_dir}\n\n"
        f"  Available runs: {avail_str}\n"
        f"  Pass the correct --run_id, or run tune.py first.\n"
    )