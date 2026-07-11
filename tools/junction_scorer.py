from langchain.tools import tool
from algorithms.score_junctions import score_junctions
from config import cfg
from tools.state import state


@tool
def junction_scorer(
    window: int | None = None,
    junction_pairs: list[list[int]] | None = None,
) -> dict:
    """Score pairwise fragment junctions using a protein language model (PLM).

    Uses masked language modeling to predict junction likelihood between each
    fragment pair. Reads fragments from shared state — call trypsin_filter first.
    The masking window can be overridden to deliberately test a different local
    context when the previous junction scores look weak: a narrower window
    tests only the immediate boundary residues (fast, but noisier on short
    motifs); a wider window gives the PLM more successor-fragment context
    (slower, but often more discriminative). A targeted subset of
    junction_pairs can be rescored and merged into the existing score matrix
    instead of recomputing every pair — use this once you know from beam_search
    diagnostics which specific junctions are weak.

    Returns score statistics over the pairs just scored (mean/min/max
    log-probability): a wide spread (max far above min) means the PLM is
    confidently distinguishing likely from unlikely junctions at this window;
    a narrow, low spread means the window isn't giving the model enough
    signal to discriminate — try widening it.
    """
    fragments = state["fragments"]
    effective_window = (
        cfg["search"]["default_levers"]["junction_window"]
        if window is None
        else int(window)
    )
    existing_scores = state.get("scores")
    unscored_pairs = state.get("unscored_junctions")
    scores = score_junctions(
        fragments,
        unscored_pairs=unscored_pairs,
        confirmed_junctions=state.get("confirmed_junctions"),
        window=effective_window,
        junction_pairs=junction_pairs,
        existing_scores=existing_scores,
    )
    state["scores"] = scores
    state["junction_window"] = effective_window

    if junction_pairs:
        scored_pairs = [(int(i), int(j)) for i, j in junction_pairs]
    elif unscored_pairs is not None:
        scored_pairs = list(unscored_pairs)
    else:
        n = len(fragments)
        scored_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    values = [scores[i, j].item() for i, j in scored_pairs]

    return {
        "num_fragments": len(fragments),
        "junction_window": state.get("junction_window"),
        "junction_pairs": junction_pairs or [],
        "merged_existing_scores": existing_scores is not None and bool(junction_pairs),
        "num_junctions_scored": len(values),
        "mean_score": sum(values) / len(values) if values else None,
        "min_score": min(values) if values else None,
        "max_score": max(values) if values else None,
    }
