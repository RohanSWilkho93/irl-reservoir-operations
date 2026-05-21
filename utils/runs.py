"""
utils/runs.py
=============
Helpers for locating and naming run folders under
results/<reservoir>/<algorithm>/.

Folder naming convention
------------------------
<run_id>_<policy_type>   e.g.  1_beta   2_lognormal   3_hardgating

run_id is always a positive integer, auto-incremented by tune.py or supplied
explicitly via --run_id.  train.py and generate_results.py use _find_run_folder
to locate an existing folder by run_id without needing to know the policy type
in advance.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Pattern that matches valid run folder names: <positive int>_<policy_type>
_RUN_FOLDER_PATTERN = re.compile(r"^(\d+)_(.+)$")


def _find_run_folder(base_dir: Path, run_id: int) -> Path:
    """
    Locate an existing run folder whose name starts with '<run_id>_'.

    Parameters
    ----------
    base_dir : Path to results/<reservoir>/<algorithm>/
    run_id   : Integer run identifier supplied via --run_id.

    Returns
    -------
    Path  The matched run folder.

    Exits with a clear error if:
      • base_dir does not exist or has no run folders at all.
      • No folder matches <run_id>_*.
      • More than one folder matches (should not happen in practice).
    """
    if not base_dir.exists():
        sys.exit(
            f"\nERROR: Results directory does not exist: {base_dir}\n"
            f"  Run tune.py for this reservoir first.\n"
        )

    matches = [
        d for d in base_dir.iterdir()
        if d.is_dir()
        and _RUN_FOLDER_PATTERN.match(d.name)
        and int(_RUN_FOLDER_PATTERN.match(d.name).group(1)) == run_id
    ]

    if not matches:
        available = sorted(
            d.name for d in base_dir.iterdir()
            if d.is_dir() and _RUN_FOLDER_PATTERN.match(d.name)
        )
        avail_str = ", ".join(available) if available else "none found"
        sys.exit(
            f"\nERROR: No run folder found for run_id={run_id} in:\n"
            f"  {base_dir}\n\n"
            f"  Available runs: {avail_str}\n"
            f"  Pass the correct --run_id, or run tune.py first.\n"
        )

    if len(matches) > 1:
        sys.exit(
            f"\nERROR: Multiple folders match run_id={run_id} in {base_dir}:\n"
            f"  {[m.name for m in matches]}\n"
            f"  This should not happen.  Remove the duplicate folder and retry.\n"
        )

    return matches[0]
