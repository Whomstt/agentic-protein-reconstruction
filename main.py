"""Single entry point for the project; behavior is controlled via config.yaml
(sweep.enabled, run.method). Usage: python main.py
"""

import multiprocessing
import os
import sys

# Windows stdout defaults to the OS codepage unless already forced to UTF-8,
# which crashes on the box-drawing characters used in progress output.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import cfg
from evaluation.runner import run_agentic, run_sequential
from evaluation.sweep import run_sweep


def main():
    # Hard guard against a combo subprocess itself starting another sweep.
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
    # freeze_support() is required on Windows since libraries using
    # multiprocessing 'spawn' re-execute this script in the child.
    multiprocessing.freeze_support()
    main()
