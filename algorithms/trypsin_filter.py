def trypsin_filter(fragments):
    """Eliminate impossible orderings based on trypsin digestion rules.

    Trypsin cleaves after K or R (unless followed by P). Therefore:
    - Every internal fragment MUST end with K or R (it was produced by a cut)
    - Only the C-terminal (last) fragment can end with a non-K/R residue
    - The N-terminal (first) fragment typically starts with M (start codon)

    Returns constraints that prune the beam search space.
    """
    n = len(fragments)

    # Fragments ending in K/R were produced by trypsin cleavage — valid in any internal position
    internal = set(i for i in range(n) if fragments[i][-1] in ("K", "R"))

    # Fragments NOT ending in K/R cannot be internal — must be last (C-terminal)
    terminal_only = set(range(n)) - internal

    # Fragments starting with M are candidates for first position (N-terminal)
    start_candidates = [i for i in range(n) if fragments[i][0] == "M"]

    # If all fragments end in K/R, any could be last (no terminal constraint)
    terminal_candidates = list(terminal_only) if terminal_only else []

    return {
        "start_candidates": start_candidates,
        "terminal_candidates": terminal_candidates,
        "internal": list(internal),
        "eliminated_from_internal": list(terminal_only),
    }
