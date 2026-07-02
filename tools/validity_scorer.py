from langchain.tools import tool

from algorithms.score_validity import pseudo_perplexity
from tools.state import state


@tool
def validity_scorer(reconstruction: str | None = None) -> float:
    """Score a reconstructed sequence with ESM-2 pseudo-perplexity.

    Lower scores indicate a more plausible protein sequence. If no explicit
    sequence is passed, the tool scores the reconstruction currently stored in
    shared state by beam_search.
    """
    sequence = (
        reconstruction
        if reconstruction is not None
        else state.get("reconstruction", "")
    )
    score = pseudo_perplexity(sequence)
    state["validity_score"] = score
    state["validity_sequence"] = sequence
    return score
