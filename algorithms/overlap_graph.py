from __future__ import annotations

from tools.sample_utils import normalize_fragment_samples


def build_overlap_graph(fragment_samples):
    """Build a peptide overlap graph from multi-sample digestions.

    The first digestion sample defines the node set. A directed edge i -> j is
    confirmed when any sampled fragment can be split into the exact
    concatenation primary[i] + primary[j]. Those confirmed edges are treated as
    hard adjacencies; all other ordered pairs among the primary fragments remain
    for scoring.
    """
    samples = normalize_fragment_samples(fragment_samples)
    primary_fragments = samples[0] if samples else []
    fragment_index = {fragment: idx for idx, fragment in enumerate(primary_fragments)}
    confirmed_junctions: set[tuple[int, int]] = set()

    for sample in samples:
        for fragment in sample:
            for split_idx in range(1, len(fragment)):
                left = fragment[:split_idx]
                right = fragment[split_idx:]
                if left in fragment_index and right in fragment_index:
                    confirmed_junctions.add(
                        (fragment_index[left], fragment_index[right])
                    )

    all_pairs = {
        (i, j)
        for i in range(len(primary_fragments))
        for j in range(len(primary_fragments))
        if i != j
    }
    unscored_junctions = sorted(all_pairs - confirmed_junctions)

    confirmed_adjacencies = [
        {
            "left_index": i,
            "right_index": j,
            "left_fragment": primary_fragments[i],
            "right_fragment": primary_fragments[j],
        }
        for i, j in sorted(confirmed_junctions)
    ]

    confirmed_successors: dict[int, set[int]] = {}
    for i, j in confirmed_junctions:
        confirmed_successors.setdefault(i, set()).add(j)

    return {
        "fragment_samples": samples,
        "fragments": primary_fragments,
        "num_samples": len(samples),
        "num_fragments": len(primary_fragments),
        "confirmed_junctions": confirmed_junctions,
        "confirmed_adjacencies": confirmed_adjacencies,
        "confirmed_successors": confirmed_successors,
        "unscored_junctions": unscored_junctions,
    }
