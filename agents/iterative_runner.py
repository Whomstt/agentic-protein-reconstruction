from __future__ import annotations

import json
import math

from config import cfg
from models.memory import free_gpu_memory, log_memory
from tools.state import state
from tools.validity_scorer import validity_scorer

TOOL_NAMES = {
    "trypsin_filter",
    "overlap_graph",
    "junction_scorer",
    "beam_search",
    "validity_scorer",
}

LEVER_KEYS = (
    "junction_window",
    "search_mode",
    "beam_width",
    "edge_mode",
    "confirmed_bonus",
)

DEFAULT_LEVER_VALUES = {
    "junction_window": cfg["mlm_model"].get("junction_window", 3),
    "search_mode": "beam",
    "beam_width": cfg["mlm_model"].get("beam_width"),
    "edge_mode": "hard",
    "confirmed_bonus": 0.0,
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
        junction_pairs = tool_summary["junction_scorer"].get("junction_pairs")
        if junction_pairs:
            pieces.append(f"rescored_pairs={len(junction_pairs)}")
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


def _effective_lever_values(tool_summary: dict, previous_values: dict | None) -> dict:
    lever_values = dict(previous_values or DEFAULT_LEVER_VALUES)

    junction_args = tool_summary.get("junction_scorer", {})
    if junction_args:
        window = junction_args.get("window")
        if window is None:
            window = DEFAULT_LEVER_VALUES["junction_window"]
        lever_values["junction_window"] = window

    beam_args = tool_summary.get("beam_search", {})
    if beam_args:
        window = beam_args.get("window")
        if window is not None:
            lever_values["junction_window"] = window
        search_mode = beam_args.get("search_mode")
        if search_mode is None:
            search_mode = DEFAULT_LEVER_VALUES["search_mode"]
        lever_values["search_mode"] = search_mode

        beam_width = beam_args.get("beam_width")
        if beam_width is None:
            beam_width = DEFAULT_LEVER_VALUES["beam_width"]
        lever_values["beam_width"] = beam_width

        edge_mode = beam_args.get("edge_mode")
        if edge_mode is None:
            edge_mode = DEFAULT_LEVER_VALUES["edge_mode"]
        lever_values["edge_mode"] = edge_mode

        confirmed_bonus = beam_args.get("confirmed_bonus")
        if confirmed_bonus is None:
            confirmed_bonus = DEFAULT_LEVER_VALUES["confirmed_bonus"]
        lever_values["confirmed_bonus"] = confirmed_bonus

    return lever_values


def _changed_levers(previous_values: dict | None, current_values: dict) -> dict:
    previous_values = previous_values or {}
    return {
        key: current_values[key]
        for key in LEVER_KEYS
        if previous_values.get(key) != current_values.get(key)
    }


def _build_iteration_prompt(iteration: int, previous_record: dict | None) -> str:
    patience = cfg["search"]["early_stop_patience"]
    max_iterations = cfg["search"]["max_iterations"]
    beam_width_step = cfg["search"]["beam_width_step"]

    if previous_record is None:
        return (
            f"Iteration {iteration}/{max_iterations}. Build an initial reconstruction hypothesis from the shared fragment sample. "
            "Use the smallest useful set of tools, but make sure you produce a candidate reconstruction and then run validity_scorer on it. "
            "The only controllable strategy levers are junction masking window, search mode, beam width, edge mode, and confirmed-edge bonus. "
            "Use junction_scorer(window=...) or beam_search(window=...) to change the masking window, beam_search(search_mode='beam'|'greedy') to change search mode, beam_search(beam_width=...) to choose any beam width, beam_search(edge_mode='hard'|'soft') to switch overlap handling, and beam_search(confirmed_bonus=...) to soften confirmed edges. "
            "If junction scores look weak, rescore a targeted subset of pairs rather than recomputing everything. "
            f"The run stops early once the best validity score (a junction+overlap plausibility score, lower is better) fails to improve for {patience} consecutive iterations, otherwise it continues until iteration {max_iterations}. "
            "If the first candidate looks weak, use a different tactic on the next iteration rather than repeating the same setup."
        )

    previous_score = previous_record["validity_score"]
    previous_strategy = previous_record.get("strategy_summary", "previous strategy")
    previous_reconstruction = previous_record.get("reconstruction", "")
    preview = previous_reconstruction[:60] + (
        "..." if len(previous_reconstruction) > 60 else ""
    )

    return (
        f"Iteration {iteration}/{max_iterations}. The previous attempt scored {previous_score:.4f} validity (junction+overlap plausibility, lower is better). "
        f"Previous strategy: {previous_strategy}. Previous reconstruction preview: {preview}. "
        "Explain briefly why that attempt likely failed, then choose a materially different tactic this round. "
        "Constraints and the overlap graph are unchanged from iteration 1 — do not call trypsin_filter or overlap_graph again; go straight to junction_scorer/beam_search. "
        "Use only these five levers: junction masking window, search mode, beam width, edge mode, and confirmed-edge bonus. "
        "If junction scores look weak, change the masking window or rescore just the suspect junction pairs. "
        f"If the search cut off early or collapsed to a partial path, change search_mode or adjust beam_width by roughly {beam_width_step} at a time. "
        "If the overlap graph and MLM disagree, switch edge_mode or adjust confirmed_bonus. "
        "Do not simply increase beam width monotonically or repeat the same setup. "
        f"The run stops early once the best validity score hasn't improved for {patience} consecutive iterations, so aim to improve on the best score seen so far. "
        "After the new candidate reconstruction, call validity_scorer again."
    )


def _extract_record(result: dict, iteration: int, prompt: str) -> dict:
    messages = result.get("messages", [])
    tool_calls = []
    tool_results = {}
    reasoning_steps = []

    for message in messages:
        msg_type = getattr(message, "type", None)

        if msg_type == "ai":
            text = getattr(message, "content", None)
            if isinstance(text, str) and text.strip():
                reasoning_steps.append(text.strip())
            calls_for_step = [
                {"name": c.get("name"), "args": c.get("args", {})}
                for c in (getattr(message, "tool_calls", None) or [])
                if c.get("name") in TOOL_NAMES
            ]
            if calls_for_step:
                reasoning_steps.append({"tool_calls": calls_for_step})

        if msg_type == "tool":
            name = getattr(message, "name", None)
            if name in TOOL_NAMES:
                reasoning_steps.append(
                    {
                        "tool_result": {
                            "name": name,
                            "content": _parse_content(
                                getattr(message, "content", None)
                            ),
                        }
                    }
                )

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
        "reasoning_steps": reasoning_steps,
        "strategy": tool_summary,
        "strategy_summary": strategy_summary,
        "lever_values": _effective_lever_values(tool_summary, None),
        "reconstruction": reconstruction,
        "order": order,
        "validity_score": validity_score,
    }


def _stream_agent(agent, prompt: str, on_event=None) -> dict:
    """Streams the agent's steps, optionally calling on_event(kind, payload) live,
    and returns a {"messages": [...]} dict compatible with _extract_record."""
    all_messages = []
    for chunk in agent.stream(
        {"messages": [("user", prompt)]},
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        if not messages:
            continue
        latest = messages[-1]
        all_messages = messages

        if on_event is None:
            continue

        msg_type = getattr(latest, "type", None)
        if msg_type == "ai":
            text = getattr(latest, "content", None)
            if isinstance(text, str) and text.strip():
                on_event("thought", text.strip())
            for call in getattr(latest, "tool_calls", None) or []:
                if call.get("name") in TOOL_NAMES:
                    on_event(
                        "tool_call",
                        {"name": call["name"], "args": call.get("args", {})},
                    )
        elif msg_type == "tool":
            name = getattr(latest, "name", None)
            if name in TOOL_NAMES:
                on_event(
                    "tool_result",
                    {
                        "name": name,
                        "content": _parse_content(getattr(latest, "content", None)),
                    },
                )

    return {"messages": all_messages}


def run_iterative_reconstruction(
    agent, fragments, fragment_samples=None, on_event=None
) -> dict:
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
    previous_levers = None
    max_iterations = cfg["search"]["max_iterations"]
    early_stop_patience = cfg["search"]["early_stop_patience"]
    # Relative margin a later iteration's validity score must clear to replace
    # the incumbent best (see algorithms/score_validity.blended_validity);
    # guards against the validity signal's noise flipping the pick to a worse
    # candidate.
    improvement_margin = cfg["search"].get("improvement_margin", 0.0)
    non_improving_streak = 0

    for iteration in range(1, max_iterations + 1):
        prompt = _build_iteration_prompt(iteration, previous_record)
        if on_event:
            on_event(
                "iteration_start",
                {"iteration": iteration, "max_iterations": max_iterations},
            )
        result = _stream_agent(agent, prompt, on_event=on_event)
        record = _extract_record(result, iteration, prompt)
        record["lever_values"] = _effective_lever_values(
            record["strategy"], previous_levers
        )
        record["changed_levers"] = _changed_levers(
            previous_levers, record["lever_values"]
        )
        history.append(record)
        state["iteration_history"] = history
        previous_levers = record["lever_values"]

        if best_record is None:
            improved = True
        else:
            # Must beat the incumbent by more than the margin to count as a real
            # improvement (guards against selecting proxy noise; see above).
            threshold = best_record["validity_score"] * (1.0 - improvement_margin)
            improved = record["validity_score"] < threshold

        if improved:
            best_record = record
            state["best_iteration"] = iteration
            state["best_reconstruction"] = record["reconstruction"]
            state["best_validity_score"] = record["validity_score"]
            state["best_order"] = record["order"]
            non_improving_streak = 0
        else:
            non_improving_streak += 1

        if on_event:
            on_event("iteration_end", record)

        free_gpu_memory()
        log_memory(f"iter {iteration}/{max_iterations}")

        if non_improving_streak >= early_stop_patience:
            break
        previous_record = record

    state["iteration_history"] = history
    return {
        "best_record": best_record or {},
        "first_record": history[0] if history else {},
        "iteration_history": history,
        "state": dict(state),
    }
