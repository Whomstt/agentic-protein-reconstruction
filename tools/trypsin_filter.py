from langchain.tools import tool
from algorithms.trypsin_filter import trypsin_filter as _trypsin_filter
from tools.state import state


@tool
def trypsin_filter(fragments: list[str]) -> dict:
    """Filter impossible fragment orderings based on trypsin digestion rules.

    Trypsin cleaves after K/R, so internal fragments must end with K or R.
    Fragments violating this are eliminated from internal positions and
    constrained to the C-terminal (last) position only. Call this first.
    """
    state.clear()
    state["fragments"] = fragments

    constraints = _trypsin_filter(fragments)
    state.update(constraints)

    n = len(fragments)
    eliminated = constraints["eliminated_from_internal"]

    return {
        "num_fragments": n,
        "start_candidates": constraints["start_candidates"],
        "terminal_candidates": constraints["terminal_candidates"],
        "eliminated_from_internal": eliminated,
        "eliminated_fragments": [fragments[i] for i in eliminated],
        "message": (
            f"{len(eliminated)}/{n} fragment(s) eliminated from internal positions "
            f"(don't end in K/R). {len(constraints['start_candidates'])} N-terminal candidate(s) (start with M)."
        ),
    }
