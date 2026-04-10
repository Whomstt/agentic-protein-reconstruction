from langchain.tools import tool
from algorithms.trypsin_filter import trypsin_filter as _trypsin_filter
from tools.state import state


@tool
def trypsin_filter(fragments: list[str]) -> dict:
    """Identify trypsin-rule junction constraints over the fragment set.

    Does NOT discard fragments. Flags adjacent-pair junctions that are
    chemically impossible:
      - K/R → P: trypsin would not have cleaved at a K/R-P site.
      - Non-K/R-ending fragments: must be C-terminal, so all outgoing
        junctions from them are impossible.
    Also marks fragments that are confirmed missed-cleavage products and
    emits an N-terminal start hint (M-starting fragments). Call this first;
    the constraints are used by beam_search to prune candidate orderings.
    """
    state.clear()
    state["fragments"] = fragments

    constraints = _trypsin_filter(fragments)
    state.update(constraints)

    n = len(fragments)
    impossible = constraints["impossible_junctions"]
    missed = constraints["missed_cleavage_fragments"]
    starts = constraints["start_candidates"]
    total_pairs = n * (n - 1)

    return {
        "num_fragments": n,
        "num_impossible_junctions": len(impossible),
        "num_total_junctions": total_pairs,
        "num_missed_cleavage_fragments": len(missed),
        "start_candidates": starts,
        "missed_cleavage_fragments": sorted(missed),
        "message": (
            f"{len(impossible)}/{total_pairs} junction(s) pruned "
            f"(K/R→P + non-K/R outgoing). "
            f"{len(missed)}/{n} fragment(s) are missed-cleavage products. "
            f"{len(starts)} N-terminal candidate(s) (M-starting)."
        ),
    }
