def trypsin_filter(fragments):
    """Identify trypsin-rule junction constraints among fragments.

    Fragments are not discarded — they have already been validated during
    preprocessing. This function produces a constraint map used to prune the
    beam search space:

    - impossible_junctions: pairs (i, j) that cannot be adjacent in the
      original sequence. Two rules contribute:
        (a) K/R → P violations — if fragment i ends in K/R and fragment j
            starts with P, trypsin would not have cleaved there.
        (b) C-terminal rule — if fragment i does not end in K/R, it must be
            the last fragment in the sequence, so (i, j) is impossible for
            every j ≠ i. Expressing the C-terminal position constraint as
            "no valid successor" keeps the output purely junction-level.
    - missed_cleavage_fragments: indices of fragments containing an internal
      K/R not followed by P — confirmed missed-cleavage products (trypsin
      skipped a legitimate cut site).
    - start_candidates: indices of fragments starting with M. This is a
      position hint (beam initialization), NOT a junction constraint, but
      it is derived from the same trypsin/biology rules and shipped here to
      avoid duplicating the fragment scan.
    """
    n = len(fragments)

    impossible_junctions = set()
    for i in range(n):
        ends_kr = fragments[i][-1] in ("K", "R")
        for j in range(n):
            if i == j:
                continue
            # C-terminal rule: non-K/R-ending fragments have no successor
            if not ends_kr:
                impossible_junctions.add((i, j))
                continue
            # Trypsin does not cleave K/R-P junctions
            if fragments[j][0] == "P":
                impossible_junctions.add((i, j))

    missed_cleavage_fragments = set()
    for i, frag in enumerate(fragments):
        for k in range(len(frag) - 1):
            if frag[k] in ("K", "R") and frag[k + 1] != "P":
                missed_cleavage_fragments.add(i)
                break

    start_candidates = [i for i in range(n) if fragments[i][0] == "M"]

    return {
        "impossible_junctions": impossible_junctions,
        "missed_cleavage_fragments": missed_cleavage_fragments,
        "start_candidates": start_candidates,
    }
