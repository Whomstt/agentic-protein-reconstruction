def trypsin_filter(fragments):
    """Identify trypsin-rule junction constraints among fragments (does not
    discard any). Returns:

    - impossible_junctions: pairs (i, j) that cannot be adjacent — either a
      K/R→P violation, or fragment i not ending in K/R (so it must be
      C-terminal and has no valid successor).
    - missed_cleavage_fragments: fragments with an internal K/R not followed
      by P (a legitimate cut site trypsin skipped).
    - start_candidates: fragments starting with M, as an N-terminal hint for
      beam initialization.
    """
    n = len(fragments)

    impossible_junctions = set()
    for i in range(n):
        ends_kr = bool(fragments[i]) and fragments[i][-1] in ("K", "R")
        for j in range(n):
            if i == j:
                continue
            # C-terminal rule: non-K/R-ending fragments have no successor
            if not ends_kr:
                impossible_junctions.add((i, j))
                continue
            # Trypsin does not cleave K/R-P junctions
            if fragments[j] and fragments[j][0] == "P":
                impossible_junctions.add((i, j))

    missed_cleavage_fragments = set()
    for i, frag in enumerate(fragments):
        for k in range(len(frag) - 1):
            if frag[k] in ("K", "R") and frag[k + 1] != "P":
                missed_cleavage_fragments.add(i)
                break

    start_candidates = [i for i in range(n) if fragments[i] and fragments[i][0] == "M"]

    return {
        "impossible_junctions": impossible_junctions,
        "missed_cleavage_fragments": missed_cleavage_fragments,
        "start_candidates": start_candidates,
    }
