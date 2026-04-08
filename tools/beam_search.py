from langchain.tools import tool
from algorithms.beam_order import beam_order
from tools.state import state


@tool
def beam_search() -> dict:
    """Find the optimal fragment ordering via beam search over junction scores.

    Uses scored junctions and trypsin constraints to reconstruct the protein.
    Reads from shared state — call junction_scorer first.
    Returns the reconstructed protein sequence and fragment order.
    """
    fragments = state["fragments"]
    order = beam_order(
        state["scores"],
        start_candidates=state.get("start_candidates"),
        terminal_candidates=state.get("terminal_candidates"),
    )
    reconstruction = "".join(fragments[i] for i in order)

    return {
        "reconstruction": reconstruction,
        "order": order,
    }
