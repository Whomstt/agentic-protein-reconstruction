from langchain.tools import tool

from algorithms.overlap_graph import build_overlap_graph
from tools.state import state


@tool
def overlap_graph(fragment_samples: list[list[str]] | list[str] | None = None) -> dict:
    """Build a multi-sample peptide overlap graph and cache hard edges.

    If no samples are passed, the tool uses fragment samples previously stored
    in shared state by trypsin_filter.
    """
    if fragment_samples is None:
        fragment_samples = state.get("fragment_samples") or [state.get("fragments", [])]

    graph = build_overlap_graph(fragment_samples)
    state.update(graph)

    return {
        "num_samples": graph["num_samples"],
        "num_fragments": graph["num_fragments"],
        "num_confirmed_adjacencies": len(graph["confirmed_adjacencies"]),
        "num_unscored_pairs": len(graph["unscored_junctions"]),
        "confirmed_adjacencies": graph["confirmed_adjacencies"],
        "message": (
            f"Built overlap graph from {graph['num_samples']} sample(s); "
            f"confirmed {len(graph['confirmed_adjacencies'])} hard adjacency edge(s) "
            f"and left {len(graph['unscored_junctions'])} ordered pair(s) for MLM scoring."
        ),
    }
