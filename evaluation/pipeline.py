from algorithms.beam_order import beam_order
from algorithms.overlap_graph import build_overlap_graph
from algorithms.score_junctions import score_junctions
from algorithms.trypsin_filter import trypsin_filter


def reconstruct(fragment_samples):
    """Run the reconstruction pipeline directly."""
    if fragment_samples and isinstance(fragment_samples[0], str):
        fragment_samples = [fragment_samples]

    fragments = fragment_samples[0]
    constraints = trypsin_filter(fragments)
    graph = build_overlap_graph(fragment_samples)
    scores = score_junctions(
        fragments,
        unscored_pairs=graph["unscored_junctions"],
        confirmed_junctions=graph["confirmed_junctions"],
    )
    order = beam_order(
        scores,
        impossible_junctions=constraints["impossible_junctions"],
        start_candidates=constraints["start_candidates"],
        confirmed_successors=graph["confirmed_successors"],
    )
    reconstruction = "".join(fragments[i] for i in order)
    return reconstruction, order, constraints, graph
