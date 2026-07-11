from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from config import cfg
from tools.trypsin_filter import trypsin_filter
from tools.overlap_graph import overlap_graph
from tools.junction_scorer import junction_scorer
from tools.beam_search import beam_search
from tools.validity_scorer import validity_scorer

TOOL_NAMES = {
    "trypsin_filter",
    "overlap_graph",
    "junction_scorer",
    "beam_search",
    "validity_scorer",
}


class LeverChoice(BaseModel):
    """The five strategy levers the agent is allowed to control."""

    reasoning: str = Field(
        description="Brief explanation of why the previous attempt likely failed and why these lever values were chosen this round."
    )
    junction_window: int = Field(
        description="Masking window (residue count) used for junction scoring."
    )
    search_mode: Literal["beam", "greedy"] = Field(
        description="Ordering search strategy."
    )
    beam_width: int = Field(description="Beam width used when search_mode is 'beam'.")
    edge_mode: Literal["hard", "soft"] = Field(
        description="How confirmed overlap-graph adjacencies constrain the search."
    )
    confirmed_bonus: float = Field(
        description="Score bonus applied to confirmed overlap-graph edges when edge_mode is 'soft'."
    )


def _default_lever_values() -> dict:
    return dict(cfg["search"]["default_levers"])


def _lever_prompt(iteration: int, previous_record: dict | None) -> str:
    max_iterations = cfg["search"]["max_iterations"]
    patience = cfg["search"]["early_stop_patience"]
    beam_width_step = cfg["search"]["beam_width_step"]

    base = (
        "You are a protein reconstruction agent choosing strategy levers for the next "
        "reconstruction attempt of unordered trypsin-digested protein fragments. "
        "You do not call tools yourself — a fixed pipeline (junction scoring, then beam/greedy "
        "search, then validity scoring) will run deterministically using the lever values you return. "
        f"This is iteration {iteration}/{max_iterations}. "
        "The five levers are: junction_window (masking window for junction scoring), "
        "search_mode ('beam' or 'greedy'), beam_width (used when search_mode='beam'), "
        "edge_mode ('hard' or 'soft' handling of overlap-graph confirmed adjacencies), and "
        "confirmed_bonus (score bonus for confirmed edges when edge_mode='soft')."
    )

    if previous_record is None:
        return (
            base
            + " This is the first iteration — there is no prior attempt to react to yet. Choose principled "
            "starting values for all five levers. A moderate junction_window (e.g. 1-5) and beam_width "
            "(e.g. 25-100) are reasonable starting points; you will get diagnostic feedback (junction score "
            "spread, beam fallback signals, confirmed-adjacency agreement) after this attempt to refine from."
        )

    previous_score = previous_record["validity_score"]
    previous_levers = previous_record.get("lever_values", {})
    previous_reconstruction = previous_record.get("reconstruction", "")
    preview = previous_reconstruction[:60] + (
        "..." if len(previous_reconstruction) > 60 else ""
    )

    diag_lines = []
    validity = previous_record.get("validity_breakdown") or {}
    j_ppl = validity.get("junction_local_ppl")
    agreement = validity.get("confirmed_adjacency_agreement")
    if j_ppl is not None:
        diag_lines.append(f"junction_local_ppl={j_ppl:.4f} (lower = more plausible junctions)")
    if agreement is not None:
        diag_lines.append(
            f"confirmed_adjacency_agreement={agreement:.2f} (fraction of real overlap-confirmed "
            "adjacencies the ordering actually placed consecutively; low = ordering ignored real evidence)"
        )

    junction_stats = previous_record.get("junction_stats") or {}
    if junction_stats.get("mean_score") is not None:
        diag_lines.append(
            f"junction scoring: mean={junction_stats['mean_score']:.4f}, "
            f"min={junction_stats['min_score']:.4f}, max={junction_stats['max_score']:.4f} "
            f"over {junction_stats['num_junctions_scored']} pairs at window={previous_levers.get('junction_window')}"
        )

    beam_diag = previous_record.get("beam_diagnostics") or {}
    if beam_diag.get("fell_back"):
        diag_lines.append(
            "beam_search fell back to a greedy extension because constraints "
            f"(beam_width={previous_levers.get('beam_width')}, edge_mode={previous_levers.get('edge_mode')}) "
            "cut off the search before covering every fragment."
        )
    if beam_diag.get("forced_impossible_count"):
        diag_lines.append(
            f"greedy search hit {beam_diag['forced_impossible_count']} step(s) where every remaining "
            "candidate was trypsin-impossible and had to pick the least-bad option anyway."
        )
    if beam_diag.get("num_confirmed_edges_total"):
        diag_lines.append(
            f"realized {beam_diag.get('num_confirmed_edges_realized', 0)}/"
            f"{beam_diag['num_confirmed_edges_total']} overlap-confirmed adjacencies as consecutive fragments."
        )

    diagnostics_text = (" Diagnostics from that attempt: " + "; ".join(diag_lines) + ".") if diag_lines else ""

    return (
        base
        + f" The previous attempt used levers {previous_levers} and scored {previous_score:.4f} validity "
        "(junction+overlap plausibility, lower is better)."
        + diagnostics_text
        + f" Previous reconstruction preview: {preview}. "
        "Choose a materially different combination of levers this round based on what the diagnostics say went wrong "
        "— don't guess blind. If junction_local_ppl is high or the junction-scoring spread is narrow, change "
        "junction_window (narrower for very local motifs, wider for more successor-fragment context). If fell_back "
        f"is true or forced_impossible_count is non-zero, change search_mode or adjust beam_width by roughly "
        f"{beam_width_step} at a time. If confirmed_adjacency_agreement is low, switch edge_mode to 'hard' (confirmed "
        "edges become a hard constraint) or, in 'soft' mode, raise confirmed_bonus so confirmed edges outscore the "
        "PLM's junction guess. Do not simply increase beam width monotonically or repeat the same setup. "
        f"The run stops early once the best validity score hasn't improved for {patience} consecutive iterations, "
        "so aim to improve on the best score seen so far."
    )


def _strategy_summary(lever_values: dict) -> str:
    return (
        f"junction_window={lever_values['junction_window']}, mode={lever_values['search_mode']}, "
        f"beam_width={lever_values['beam_width']}, edge_mode={lever_values['edge_mode']}, "
        f"confirmed_bonus={lever_values['confirmed_bonus']}"
    )


def _emit_tool_step(on_event, name: str, args: dict, result) -> dict:
    if on_event:
        on_event("tool_call", {"name": name, "args": args})
        on_event("tool_result", {"name": name, "content": result})
    return {
        "tool_call": {"name": name, "args": args},
        "tool_result": {"name": name, "content": result},
    }


def run_single_call_iteration(
    llm, iteration: int, previous_record: dict | None, on_event=None
) -> dict:
    """Runs one iteration in single-call mode: one LLM call to pick the five
    levers, then a deterministic tool pipeline executed directly in Python."""
    reasoning_steps: list = []

    if iteration == 1:
        trypsin_result = trypsin_filter.invoke({})
        reasoning_steps.append(
            {"tool_result": {"name": "trypsin_filter", "content": trypsin_result}}
        )
        if on_event:
            on_event("tool_call", {"name": "trypsin_filter", "args": {}})
            on_event(
                "tool_result", {"name": "trypsin_filter", "content": trypsin_result}
            )

        overlap_result = overlap_graph.invoke({})
        reasoning_steps.append(
            {"tool_result": {"name": "overlap_graph", "content": overlap_result}}
        )
        if on_event:
            on_event("tool_call", {"name": "overlap_graph", "args": {}})
            on_event(
                "tool_result", {"name": "overlap_graph", "content": overlap_result}
            )

    iteration1_deterministic = cfg["run"].get("iteration1_deterministic", True)
    use_llm_for_levers = iteration > 1 or not iteration1_deterministic

    if iteration == 1 and not use_llm_for_levers:
        # run.iteration1_deterministic is true: iteration 1 uses the fixed
        # config defaults directly, no LLM call. The LLM only decides lever
        # values for iterations 2+, once it has a real previous-attempt
        # score/strategy to react to. This trades one LLM call/sample for a
        # weaker claim: "1st Iteration" in reports means "config defaults",
        # not "the LLM's own first attempt" — see CLAUDE.md.
        prompt = None
        lever_values = _default_lever_values()
        reasoning_note = "Iteration 1: using config default levers (no LLM call; run.iteration1_deterministic=true)."
        reasoning_steps.append(reasoning_note)
        if on_event:
            on_event("thought", reasoning_note)
    else:
        prompt = _lever_prompt(iteration, previous_record)
        structured_llm = llm.with_structured_output(LeverChoice)
        choice: LeverChoice = structured_llm.invoke(prompt)

        if on_event and choice.reasoning:
            on_event("thought", choice.reasoning.strip())
        if choice.reasoning:
            reasoning_steps.append(choice.reasoning.strip())

        lever_values = {
            "junction_window": choice.junction_window,
            "search_mode": choice.search_mode,
            "beam_width": choice.beam_width,
            "edge_mode": choice.edge_mode,
            "confirmed_bonus": choice.confirmed_bonus,
        }

    previous_window = (
        previous_record.get("lever_values", {}).get("junction_window")
        if previous_record
        else None
    )
    window_unchanged = iteration > 1 and lever_values["junction_window"] == previous_window

    junction_args = {"window": lever_values["junction_window"]}
    if window_unchanged:
        # Junction scores for this window are already cached in shared state
        # from the previous iteration; skip the full MLM rescore. beam_search
        # below will see the matching window and reuse state["scores"] itself.
        note = f"junction_window unchanged ({lever_values['junction_window']}) — reusing cached junction scores, skipping rescore."
        reasoning_steps.append(note)
        if on_event:
            on_event("thought", note)
        junction_stats = (previous_record or {}).get("junction_stats")
    else:
        junction_result = junction_scorer.invoke(junction_args)
        reasoning_steps.append(
            {"tool_call": {"name": "junction_scorer", "args": junction_args}}
        )
        reasoning_steps.append(
            {"tool_result": {"name": "junction_scorer", "content": junction_result}}
        )
        if on_event:
            on_event("tool_call", {"name": "junction_scorer", "args": junction_args})
            on_event(
                "tool_result", {"name": "junction_scorer", "content": junction_result}
            )
        junction_stats = {
            "mean_score": junction_result.get("mean_score"),
            "min_score": junction_result.get("min_score"),
            "max_score": junction_result.get("max_score"),
            "num_junctions_scored": junction_result.get("num_junctions_scored"),
        }

    beam_args = {
        "search_mode": lever_values["search_mode"],
        "beam_width": lever_values["beam_width"],
        "edge_mode": lever_values["edge_mode"],
        "confirmed_bonus": lever_values["confirmed_bonus"],
        "window": lever_values["junction_window"],
    }
    beam_result = beam_search.invoke(beam_args)
    reasoning_steps.append({"tool_call": {"name": "beam_search", "args": beam_args}})
    reasoning_steps.append(
        {"tool_result": {"name": "beam_search", "content": beam_result}}
    )
    if on_event:
        on_event("tool_call", {"name": "beam_search", "args": beam_args})
        on_event("tool_result", {"name": "beam_search", "content": beam_result})

    beam_diagnostics = {
        "fell_back": beam_result.get("fell_back"),
        "forced_impossible_count": beam_result.get("forced_impossible_count"),
        "num_confirmed_edges_realized": beam_result.get("num_confirmed_edges_realized"),
        "num_confirmed_edges_total": beam_result.get("num_confirmed_edges_total"),
        "mean_junction_score": beam_result.get("mean_junction_score"),
    }

    validity_result = validity_scorer.invoke({})
    reasoning_steps.append({"tool_call": {"name": "validity_scorer", "args": {}}})
    reasoning_steps.append(
        {"tool_result": {"name": "validity_scorer", "content": validity_result}}
    )
    if on_event:
        on_event("tool_call", {"name": "validity_scorer", "args": {}})
        on_event(
            "tool_result", {"name": "validity_scorer", "content": validity_result}
        )

    validity_breakdown = validity_result if isinstance(validity_result, dict) else {}
    try:
        validity_score = float(validity_breakdown.get("validity_score", validity_result))
    except (TypeError, ValueError):
        validity_score = float("inf")

    return {
        "iteration": iteration,
        "prompt": prompt,
        "reasoning_steps": reasoning_steps,
        "strategy": {"junction_scorer": junction_args, "beam_search": beam_args},
        "strategy_summary": _strategy_summary(lever_values),
        "lever_values": lever_values,
        "reconstruction": beam_result.get("reconstruction", ""),
        "order": beam_result.get("order", []),
        "validity_score": validity_score,
        "validity_breakdown": validity_breakdown,
        "junction_stats": junction_stats,
        "beam_diagnostics": beam_diagnostics,
        "llm_call": use_llm_for_levers,
    }
