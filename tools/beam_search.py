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
    sequence and fragment order. The agent can switch between greedy and beam
    search, widen the beam, or soften confirmed overlap edges to probe a
    different reconstruction hypothesis.
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
            cfg["mlm_model"].get("junction_window", 3)
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

    if search_mode == "greedy":
        order = greedy_order(
            state["scores"],
            impossible_junctions=state.get("impossible_junctions"),
            start_candidates=state.get("start_candidates"),
            confirmed_successors=state.get("confirmed_successors"),
            edge_mode=edge_mode,
            confirmed_bonus=confirmed_bonus,
        )
    else:
        order = beam_order(
            state["scores"],
            impossible_junctions=state.get("impossible_junctions"),
            start_candidates=state.get("start_candidates"),
            confirmed_successors=state.get("confirmed_successors"),
            beam_size=beam_width,
            edge_mode=edge_mode,
            confirmed_bonus=confirmed_bonus,
        )
    reconstruction = "".join(fragments[i] for i in order)
    state["reconstruction"] = reconstruction
    state["order"] = order

    return {
        "reconstruction": reconstruction,
        "order": order,
        "search_mode": search_mode,
        "beam_width": beam_width,
        "edge_mode": edge_mode,
        "confirmed_bonus": confirmed_bonus,
    }
