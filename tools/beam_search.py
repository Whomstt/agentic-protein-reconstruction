from langchain.tools import tool
from algorithms.beam_order import beam_order
from algorithms.overlap_graph import build_overlap_graph
from algorithms.score_junctions import score_junctions
from algorithms.trypsin_filter import trypsin_filter as _trypsin_filter
from tools.state import state


@tool
def beam_search() -> dict:
    """Find the optimal fragment ordering via beam search over junction scores.

    Uses scored junctions, the trypsin impossible-junction set, and the
    N-terminal start hint to reconstruct the protein. Reads from shared
    state and will lazily compute any missing prerequisites so the agent can
    skip intermediate tools when appropriate. Returns the reconstructed protein
    sequence and fragment order.
    """
    fragments = state["fragments"]

    if "impossible_junctions" not in state or "start_candidates" not in state:
        constraints = _trypsin_filter(fragments)
        state.update(constraints)

    if "confirmed_junctions" not in state or "unscored_junctions" not in state:
        fragment_samples = state.get("fragment_samples") or [fragments]
        graph = build_overlap_graph(fragment_samples)
        state.update(graph)

    if "scores" not in state:
        scores = score_junctions(
            fragments,
            unscored_pairs=state.get("unscored_junctions"),
            confirmed_junctions=state.get("confirmed_junctions"),
        )
        state["scores"] = scores

    order = beam_order(
        state["scores"],
        impossible_junctions=state.get("impossible_junctions"),
        start_candidates=state.get("start_candidates"),
        confirmed_successors=state.get("confirmed_successors"),
    )
    reconstruction = "".join(fragments[i] for i in order)

    return {
        "reconstruction": reconstruction,
        "order": order,
    }
