from langchain.tools import tool
from algorithms.score_junctions import score_junctions
from config import cfg
from tools.state import state


@tool
def junction_scorer(
    window: int | None = None,
    junction_pairs: list[list[int]] | None = None,
) -> dict:
    """Score all pairwise fragment junctions using a protein language model (PLM).

    Uses masked language modeling to predict junction likelihood between each
    fragment pair. Reads fragments from shared state — call trypsin_filter first.
    The masking window can be overridden to deliberately test a different local
    context when the previous junction scores look weak. A targeted subset of
    junction pairs can be rescored and merged into the existing score matrix.
    """
    fragments = state["fragments"]
    effective_window = (
        cfg["mlm_model"].get("junction_window", 3) if window is None else int(window)
    )
    existing_scores = state.get("scores")
    scores = score_junctions(
        fragments,
        unscored_pairs=state.get("unscored_junctions"),
        confirmed_junctions=state.get("confirmed_junctions"),
        window=effective_window,
        junction_pairs=junction_pairs,
        existing_scores=existing_scores,
    )
    state["scores"] = scores
    state["junction_window"] = effective_window

    return {
        "num_fragments": len(fragments),
        "junction_window": state.get("junction_window"),
        "junction_pairs": junction_pairs or [],
        "merged_existing_scores": existing_scores is not None and bool(junction_pairs),
        "scores_computed": True,
    }
