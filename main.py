"""Single entry point for the project. Everything is controlled via config.yaml:

- sweep.enabled: false (default) -> runs once using run.method below.
  sweep.enabled: true             -> loops through sweep.grid (+ extra_runs),
                                      each combo running this same file again
                                      as a subprocess.
- run.method: "agentic" (default) or "sequential" -> which reconstruction
  approach to evaluate when not sweeping.

Before every non-sweep run, the fragmented dataset for the active organism is
regenerated automatically if data.organism, data.replica_count, or
data.missed_cleavage_ratio have changed since it was last written (see
preprocessing.preprocessing.ensure_fresh_dataset). Each sweep combo runs this
same file as its own subprocess against a per-combo config, so this check
naturally re-runs preprocessing once per distinct (organism, replica_count,
missed_cleavage_ratio) combination in the grid.

Usage: python main.py
"""

import multiprocessing
import os
import sys

# Console output uses box-drawing characters (progress bars, dividers). On
# Windows, stdout defaults to the OS codepage (cp1252) unless the terminal or
# PYTHONIOENCODING already forces UTF-8, which crashes the first time one of
# those characters gets printed. Force it here so this runs the same way
# regardless of terminal/codepage, including when spawned as a sweep combo's
# subprocess.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import cfg
from evaluation.runner import run_agentic, run_sequential
from evaluation.sweep import run_sweep


def main():
    # A combo subprocess must never itself start a sweep. Its per-combo config
    # already sets sweep.enabled=false, but run_sweep also exports
    # AGENTIC_IN_SWEEP_CHILD=1 into the child env as a hard guard: if anything
    # (a misread config, or a stray multiprocessing re-exec) reached here inside
    # a combo child, this stops it from spawning yet another sweep and cascading
    # into the process/memory explosion that used to OOM the machine.
    in_sweep_child = os.environ.get("AGENTIC_IN_SWEEP_CHILD") == "1"

    if cfg.get("sweep", {}).get("enabled") and not in_sweep_child:
        run_sweep(cfg["sweep"])
        return

    from preprocessing.preprocessing import ensure_fresh_dataset

    ensure_fresh_dataset()

    method = cfg.get("run", {}).get("method", "agentic")
    if method == "sequential":
        run_sequential()
    elif method == "agentic":
        run_agentic()
    else:
        raise ValueError(
            f"Unknown run.method '{method}' in config.yaml. Expected 'agentic' or 'sequential'."
        )


if __name__ == "__main__":
    # On Windows, any library that uses multiprocessing 'spawn' re-executes this
    # script in the child. freeze_support() makes that child return immediately
    # instead of falling through and kicking off another full sweep.
    multiprocessing.freeze_support()
    main()
