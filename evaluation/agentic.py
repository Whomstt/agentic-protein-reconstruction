"""CLI wrapper for the agentic evaluation path, independent of config.yaml's
run.method (which main.py uses to pick between agentic/sequential). All
actual logic lives in evaluation/runner.py.

Usage: python -m evaluation.agentic
"""

from evaluation.runner import run_agentic

if __name__ == "__main__":
    run_agentic()
