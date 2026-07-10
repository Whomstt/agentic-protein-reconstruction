"""Run the iterative agent on a SINGLE sample in its own process, write the
result as JSON, and exit.

evaluation/runner.run_agentic spawns one of these per sample. The LLM stack
(LangChain / LangGraph / OpenAI-httpx / Azure) grows native, non-Python host
memory that only a process exit reclaims — proven by measurement: tools, torch,
data and the process tree were all flat, and rebuilding the agent object in-process
did not help, but a single 5-sample combo still climbed past 24 GB. Isolating each
sample in a short-lived subprocess keeps the long-running combo process flat: it
only aggregates lightweight JSON results, while each worker starts fresh (~1.3 GB)
and frees everything on exit.

Usage: python -m evaluation.sample_worker <sample_json_in> <result_json_out>
"""

import json
import multiprocessing
import sys


# Match main.py: the agent's live-reasoning output uses box-drawing characters,
# which crash on Windows' default cp1252 stdout. Force UTF-8 before anything prints.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _json_safe(obj):
    """Recursively coerce to JSON-serializable types (sets/tuples -> lists,
    anything exotic -> str) so the result always round-trips to the parent."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int, float)):
        return obj
    return str(obj)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    sample_path, output_path = sys.argv[1], sys.argv[2]
    with open(sample_path, encoding="utf-8") as f:
        sample = json.load(f)

    fragment_samples = sample.get("fragment_samples") or [sample["fragments"]]
    fragments = fragment_samples[0]

    from agents.iterative_runner import run_iterative_reconstruction
    from agents.react_agent import build_agent
    from evaluation.runner import make_event_printer

    agent = build_agent()
    run_result = run_iterative_reconstruction(
        agent, fragments, fragment_samples, on_event=make_event_printer()
    )

    # Keep only the fields run_agentic consumes, made JSON-safe.
    state = run_result.get("state", {})
    payload = {
        "best_record": _json_safe(run_result.get("best_record", {})),
        "first_record": _json_safe(run_result.get("first_record", {})),
        "iteration_history": _json_safe(run_result.get("iteration_history", [])),
        "state": {
            "impossible_junctions": _json_safe(
                list(state.get("impossible_junctions", []))
            ),
            "confirmed_adjacencies": _json_safe(
                state.get("confirmed_adjacencies", [])
            ),
            "unscored_junctions": _json_safe(state.get("unscored_junctions", [])),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
