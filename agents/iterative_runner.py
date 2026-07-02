from __future__ import annotations

import json
import math

from config import cfg
from tools.state import state
from tools.validity_scorer import validity_scorer

TOOL_NAMES = {
    "trypsin_filter",
    "overlap_graph",
    "junction_scorer",
    "beam_search",
    "validity_scorer",
}


def _parse_content(content):
    if isinstance(content, (dict, list, int, float)):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                return float(content)
            except ValueError:
                return content
    return content


def _tool_call_summary(tool_calls: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for call in tool_calls:
        name = call.get("name")
        if name in TOOL_NAMES:
            summary[name] = call.get("args", {}) or {}
    return summary


def _strategy_summary(tool_summary: dict) -> str:
    pieces = []
    if "junction_scorer" in tool_summary:
        window = tool_summary["junction_scorer"].get("window")
        if window is not None:
            pieces.append(f"junction_window={window}")
    if "beam_search" in tool_summary:
        beam_args = tool_summary["beam_search"]
        mode = beam_args.get("search_mode")
        beam_width = beam_args.get("beam_width")
        edge_mode = beam_args.get("edge_mode")
        confirmed_bonus = beam_args.get("confirmed_bonus")
        if mode is not None:
            pieces.append(f"mode={mode}")
        if beam_width is not None:
            pieces.append(f"beam_width={beam_width}")
        if edge_mode is not None:
            pieces.append(f"edge_mode={edge_mode}")
        if confirmed_bonus not in (None, 0, 0.0):
            pieces.append(f"confirmed_bonus={confirmed_bonus}")
    if not pieces:
        return "default reconstruction path"
    return ", ".join(pieces)


def _build_iteration_prompt(iteration: int, previous_record: dict | None) -> str:
    threshold = cfg["search"]["validity_threshold"]
    max_iterations = cfg["search"]["max_iterations"]
    beam_step = cfg["search"]["beam_width_step"]

    if previous_record is None:
        return (
            f"Iteration {iteration}/{max_iterations}. Build an initial reconstruction hypothesis from the shared fragment sample. "
            "Use the smallest useful set of tools, but make sure you produce a candidate reconstruction and then run validity_scorer on it. "
            f"The validity target is pseudo-perplexity <= {threshold}. "
            "If the first candidate looks weak, use a different tactic on the next iteration rather than repeating the same setup."
        )

    previous_score = previous_record["validity_score"]
    previous_strategy = previous_record.get("strategy_summary", "previous strategy")
    previous_reconstruction = previous_record.get("reconstruction", "")
    preview = previous_reconstruction[:60] + (
        "..." if len(previous_reconstruction) > 60 else ""
    )

    return (
        f"Iteration {iteration}/{max_iterations}. The previous attempt scored {previous_score:.4f} pseudo-perplexity, which is above the target of {threshold}. "
        f"Previous strategy: {previous_strategy}. Previous reconstruction preview: {preview}. "
        f"Explain briefly why that attempt likely failed, then choose a materially different tactic this round. Do not simply increase beam width; change the reasoning path. "
        f"Useful moves include rerunning junction_scorer with a different window, switching beam_search to greedy, increasing or decreasing beam_width by {beam_step}, or softening confirmed overlap edges with edge_mode='soft'. "
        "After the new candidate reconstruction, call validity_scorer again."
    )


def _extract_record(result: dict, iteration: int, prompt: str) -> dict:
    messages = result.get("messages", [])
    tool_calls = []
    tool_results = {}

    for message in messages:
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls.extend(message.tool_calls)
        name = getattr(message, "name", None)
        if name in TOOL_NAMES:
            tool_results[name] = _parse_content(getattr(message, "content", None))

    tool_summary = _tool_call_summary(tool_calls)
    strategy_summary = _strategy_summary(tool_summary)

    beam_result = tool_results.get("beam_search", {})
    if not isinstance(beam_result, dict):
        beam_result = {}

    reconstruction = beam_result.get("reconstruction", state.get("reconstruction", ""))
    order = beam_result.get("order", state.get("order", []))

    validity_value = tool_results.get("validity_scorer")
    if isinstance(validity_value, dict):
        validity_value = validity_value.get("validity_score")
    if validity_value is None and reconstruction:
        validity_value = validity_scorer(reconstruction)

    try:
        validity_score = float(validity_value)
    except (TypeError, ValueError):
        validity_score = float("inf")

    if math.isnan(validity_score):
        validity_score = float("inf")

    return {
        "iteration": iteration,
        "prompt": prompt,
        "strategy": tool_summary,
        "strategy_summary": strategy_summary,
        "reconstruction": reconstruction,
        "order": order,
        "validity_score": validity_score,
    }


def run_iterative_reconstruction(agent, fragments, fragment_samples=None) -> dict:
    state.clear()
    state["fragment_samples"] = fragment_samples or [fragments]
    state["fragments"] = fragments
    state["iteration_history"] = []
    state["best_reconstruction"] = ""
    state["best_validity_score"] = float("inf")
    state["best_order"] = []

    history = []
    best_record = None
    previous_record = None
    max_iterations = cfg["search"]["max_iterations"]
    threshold = cfg["search"]["validity_threshold"]

    for iteration in range(1, max_iterations + 1):
        prompt = _build_iteration_prompt(iteration, previous_record)
        result = agent.invoke({"messages": [("user", prompt)]})
        record = _extract_record(result, iteration, prompt)
        history.append(record)
        state["iteration_history"] = history

        if (
            best_record is None
            or record["validity_score"] < best_record["validity_score"]
        ):
            best_record = record
            state["best_iteration"] = iteration
            state["best_reconstruction"] = record["reconstruction"]
            state["best_validity_score"] = record["validity_score"]
            state["best_order"] = record["order"]

        if record["validity_score"] <= threshold:
            break

        previous_record = record

    state["iteration_history"] = history
    return {
        "best_record": best_record or {},
        "iteration_history": history,
        "state": dict(state),
    }
