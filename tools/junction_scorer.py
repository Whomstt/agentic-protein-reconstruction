from langchain.tools import tool
from algorithms.score_junctions import score_junctions
from tools.state import state


@tool
def junction_scorer() -> dict:
    """Score all pairwise fragment junctions using a protein language model (PLM).

    Uses masked language modeling to predict junction likelihood between each
    fragment pair. Reads fragments from shared state — call trypsin_filter first.
    """
    fragments = state["fragments"]
    scores = score_junctions(fragments)
    state["scores"] = scores

    return {
        "num_fragments": len(fragments),
        "scores_computed": True,
    }
