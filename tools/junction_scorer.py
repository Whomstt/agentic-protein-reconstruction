from langchain.tools import tool
from algorithms.score_junctions import score_junctions
from tools.state import state


@tool
def junction_scorer(window: int | None = None) -> dict:
    """Score all pairwise fragment junctions using a protein language model (PLM).

    Uses masked language modeling to predict junction likelihood between each
    fragment pair. Reads fragments from shared state — call trypsin_filter first.
    The masking window can be overridden to deliberately test a different local
    context when the previous junction scores look weak.
    """
    fragments = state["fragments"]
    scores = score_junctions(
        fragments,
        unscored_pairs=state.get("unscored_junctions"),
        confirmed_junctions=state.get("confirmed_junctions"),
        window=window,
    )
    state["scores"] = scores
    state["junction_window"] = (
        window if window is not None else state.get("junction_window")
    )

    return {
        "num_fragments": len(fragments),
        "junction_window": state.get("junction_window"),
        "scores_computed": True,
    }
