"""CLI wrapper for the deterministic pipeline evaluation path, independent of
config.yaml's run.method (which main.py uses to pick between
agentic/sequential). All actual logic lives in evaluation/runner.py.

Usage: python -m evaluation.sequential
"""

from evaluation.runner import run_sequential

if __name__ == "__main__":
    run_sequential()
