from langchain.tools import tool
from algorithms.trypsin_filter import trypsin_filter as _trypsin_filter
from tools.sample_utils import normalize_fragment_samples, primary_fragments
from tools.state import state


@tool
def trypsin_filter(fragments: list[str] | list[list[str]] | None = None) -> dict:
    """Identify trypsin-rule junction constraints over the fragment set.

    Does NOT discard fragments. Flags adjacent-pair junctions that are
    chemically impossible:
      - K/R → P: trypsin would not have cleaved at a K/R-P site.
      - Non-K/R-ending fragments: must be C-terminal, so all outgoing
        junctions from them are impossible.
    Also marks fragments that are confirmed missed-cleavage products and
    emits an N-terminal start hint (M-starting fragments). If no fragments are
    passed, the tool uses the fragment sample already stored in shared state.
    """
    fragment_samples = normalize_fragment_samples(
        fragments if fragments is not None else state.get("fragment_samples")
    )
    primary = primary_fragments(fragment_samples)

    if state.get("fragments") == primary and "impossible_junctions" in state:
        constraints = {
            "impossible_junctions": state["impossible_junctions"],
            "missed_cleavage_fragments": state["missed_cleavage_fragments"],
            "start_candidates": state["start_candidates"],
        }
    else:
        state["fragment_samples"] = fragment_samples or [primary]
        state["fragments"] = primary
        constraints = _trypsin_filter(primary)
        state.update(constraints)

    n = len(primary)
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
