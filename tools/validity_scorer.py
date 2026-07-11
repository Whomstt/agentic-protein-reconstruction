from langchain.tools import tool

from algorithms.score_validity import blended_validity_detailed, pseudo_perplexity
from config import cfg
from tools.state import state


@tool
def validity_scorer(reconstruction: str | None = None) -> dict:
    """Score a reconstructed ordering as the basis for best-candidate selection.

    Returns a breakdown, not just one number, so the next iteration's lever
    choice can target the actual weak point instead of guessing:
      - validity_score: the blended score (lower is better); this is what
        best-candidate selection compares across iterations.
      - junction_local_ppl: masked-LM plausibility of the ordering's
        non-confirmed junctions. High => the PLM finds these junctions
        implausible — try a different junction_window (narrower for very
        local motifs, wider for more context) or rescore with beam_search's
        window argument.
      - confirmed_adjacency_agreement: fraction (0-1, or null if the overlap
        graph has no confirmed edges) of overlap-graph confirmed adjacencies
        the ordering actually realized as consecutive fragments. Low =>
        the search ignored real multi-replica overlap evidence — switch
        edge_mode to 'hard' (confirmed edges become hard constraints) or, in
        'soft' mode, raise confirmed_bonus so confirmed edges outscore the
        PLM's junction guess.
      - confirmed_penalty_applied: how much confirmed_adjacency_agreement
        actually inflated validity_score this call (0 when agreement is
        perfect or there are no confirmed edges to violate).

    Scores the ordering currently in shared state (set by beam_search). If an
    explicit reconstruction string different from state's is passed, the tool
    falls back to whole-sequence pseudo-perplexity on that string (its fragment
    ordering isn't known) and only validity_score is meaningful.
    """
    fragments = state.get("fragments") or []
    state_recon = state.get("reconstruction", "")
    sequence = reconstruction if reconstruction is not None else state_recon

    # Only use the junction+overlap blend when we know the ordering, i.e. when
    # scoring the reconstruction beam_search already stored in state.
    order = state.get("order") if (reconstruction is None or reconstruction == state_recon) else None

    if order and fragments:
        result = blended_validity_detailed(
            fragments,
            order,
            state.get("confirmed_junctions"),
            cfg["search"].get("validity_junction_window", 5),
            cfg["search"].get("validity_confirmed_penalty", 0.75),
        )
    else:
        result = {
            "validity_score": pseudo_perplexity(sequence),
            "junction_local_ppl": None,
            "confirmed_adjacency_agreement": None,
            "confirmed_penalty_applied": 0.0,
        }

    state["validity_score"] = result["validity_score"]
    state["validity_sequence"] = sequence
    return result
