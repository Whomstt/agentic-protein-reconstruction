from langchain.tools import tool

from algorithms.score_validity import blended_validity, pseudo_perplexity
from config import cfg
from tools.state import state


@tool
def validity_scorer(reconstruction: str | None = None) -> float:
    """Score a reconstructed ordering as the basis for best-candidate selection.

    Lower is better. Combines junction-local pseudo-perplexity (masked-LM
    plausibility of just the fragment boundaries the ordering uses) with
    confirmed-adjacency agreement (how well the ordering respects the overlap
    graph's confirmed adjacencies).

    Scores the ordering currently in shared state (set by beam_search). If an
    explicit reconstruction string different from state's is passed, the tool
    falls back to whole-sequence pseudo-perplexity on that string (its fragment
    ordering isn't known).
    """
    fragments = state.get("fragments") or []
    state_recon = state.get("reconstruction", "")
    sequence = reconstruction if reconstruction is not None else state_recon

    # Only use the junction+overlap blend when we know the ordering, i.e. when
    # scoring the reconstruction beam_search already stored in state.
    order = state.get("order") if (reconstruction is None or reconstruction == state_recon) else None

    if order and fragments:
        score = blended_validity(
            fragments,
            order,
            state.get("confirmed_junctions"),
            cfg["search"].get("validity_junction_window", 5),
            cfg["search"].get("validity_confirmed_penalty", 0.75),
        )
    else:
        score = pseudo_perplexity(sequence)

    state["validity_score"] = score
    state["validity_sequence"] = sequence
    return score
