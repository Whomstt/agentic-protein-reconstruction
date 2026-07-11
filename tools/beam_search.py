from langchain.tools import tool
from algorithms.beam_order import beam_order
from algorithms.greedy_order import greedy_order
from algorithms.overlap_graph import build_overlap_graph
from algorithms.score_junctions import score_junctions
from algorithms.trypsin_filter import trypsin_filter as _trypsin_filter
from config import cfg
from tools.state import state


@tool
def beam_search(
    search_mode: str = "beam",
    beam_width: int | None = None,
    edge_mode: str = "hard",
    confirmed_bonus: float = 0.0,
    window: int | None = None,
) -> dict:
    """Find the optimal fragment ordering via beam search over junction scores.

    Uses scored junctions, the trypsin impossible-junction set, and the
    N-terminal start hint to reconstruct the protein. Reads from shared
    state and will lazily compute any missing prerequisites so the agent can
    skip intermediate tools when appropriate. Returns the reconstructed protein
    sequence and fragment order, plus search diagnostics to guide the next
    lever choice:
      - fell_back (search_mode='beam' only): true if constraints (impossible
        junctions / edge_mode='hard' confirmed successors) cut off the beam
        before it covered every fragment, forcing a greedy extension of the
        best partial beam. A true here means beam_width or edge_mode is too
        restrictive for this fragment set — widen beam_width or try
        edge_mode='soft'.
      - forced_impossible_count (search_mode='greedy' only): how many steps
        had every remaining candidate marked impossible and had to pick the
        least-bad one anyway. Non-zero means the trypsin constraints are
        fighting the ordering — try search_mode='beam', which explores
        multiple candidate paths instead of committing greedily.
      - num_confirmed_edges_realized / num_confirmed_edges_total: how many of
        the overlap graph's confirmed adjacencies this ordering actually
        placed as consecutive fragments. A low ratio (and edge_mode='soft')
        suggests raising confirmed_bonus, or switching to edge_mode='hard' to
        force those edges.
      - mean_junction_score: average raw MLM score across this ordering's
        consecutive junctions (higher/less-negative is more plausible) —
        a cheap preview of validity_scorer's junction_local_ppl before
        paying for the full validity computation.

    The agent can switch between greedy and beam search, widen the beam, or
    soften/harden confirmed overlap edges to probe a different reconstruction
    hypothesis; see the diagnostics above for which lever is likely to help.
    """
    fragments = state["fragments"]
    state["search_strategy"] = {
        "search_mode": search_mode,
        "beam_width": beam_width,
        "edge_mode": edge_mode,
        "confirmed_bonus": confirmed_bonus,
        "window": window,
    }

    if "impossible_junctions" not in state or "start_candidates" not in state:
        constraints = _trypsin_filter(fragments)
        state.update(constraints)

    if "confirmed_junctions" not in state or "unscored_junctions" not in state:
        fragment_samples = state.get("fragment_samples") or [fragments]
        graph = build_overlap_graph(fragment_samples)
        state.update(graph)

    needs_rescore = "scores" not in state or (
        window is not None and state.get("junction_window") != window
    )
    if needs_rescore:
        effective_window = (
            cfg["search"]["default_levers"]["junction_window"]
            if window is None
            else int(window)
        )
        scores = score_junctions(
            fragments,
            unscored_pairs=state.get("unscored_junctions"),
            confirmed_junctions=state.get("confirmed_junctions"),
            window=effective_window,
        )
        state["scores"] = scores
        state["junction_window"] = effective_window

    diagnostics: dict = {}
    if search_mode == "greedy":
        order = greedy_order(
            state["scores"],
            impossible_junctions=state.get("impossible_junctions"),
            start_candidates=state.get("start_candidates"),
            confirmed_successors=state.get("confirmed_successors"),
            edge_mode=edge_mode,
            confirmed_bonus=confirmed_bonus,
            diagnostics=diagnostics,
        )
    else:
        order = beam_order(
            state["scores"],
            impossible_junctions=state.get("impossible_junctions"),
            start_candidates=state.get("start_candidates"),
            confirmed_successors=state.get("confirmed_successors"),
            beam_width=beam_width,
            edge_mode=edge_mode,
            confirmed_bonus=confirmed_bonus,
            diagnostics=diagnostics,
        )
    reconstruction = "".join(fragments[i] for i in order)
    state["reconstruction"] = reconstruction
    state["order"] = order

    consecutive_edges = {(order[k], order[k + 1]) for k in range(len(order) - 1)}
    confirmed = state.get("confirmed_junctions") or set()
    num_confirmed_realized = sum(1 for edge in confirmed if edge in consecutive_edges)

    scores = state["scores"]
    junction_scores = [
        scores[order[k], order[k + 1]].item() for k in range(len(order) - 1)
    ]
    mean_junction_score = (
        sum(junction_scores) / len(junction_scores) if junction_scores else None
    )

    return {
        "reconstruction": reconstruction,
        "order": order,
        "search_mode": search_mode,
        "beam_width": beam_width,
        "edge_mode": edge_mode,
        "confirmed_bonus": confirmed_bonus,
        "fell_back": diagnostics.get("fell_back"),
        "forced_impossible_count": diagnostics.get("forced_impossible_count"),
        "num_confirmed_edges_realized": num_confirmed_realized,
        "num_confirmed_edges_total": len(confirmed),
        "mean_junction_score": mean_junction_score,
    }
