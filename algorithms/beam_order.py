from config import cfg


def beam_order(
    scores,
    impossible_junctions=None,
    start_candidates=None,
    confirmed_successors=None,
    beam_width=None,
    edge_mode="hard",
    confirmed_bonus=0.0,
    diagnostics: dict | None = None,
):
    """diagnostics, if passed, is populated in place with:
      - fell_back: whether constraints cut off the beam before it covered
        every fragment, forcing a greedy extension of the best partial beam.
      - fragments_placed_by_beam: how many fragments the beam itself placed
        before any fallback extension kicked in.
    """
    n = scores.shape[0]
    beam_width = (
        cfg["search"]["default_levers"]["beam_width"]
        if beam_width is None
        else beam_width
    )
    impossible = impossible_junctions or set()
    confirmed_successors = confirmed_successors or {}

    if start_candidates:
        beams = [(0.0, [i], {i}) for i in start_candidates]
    else:
        beams = [(0.0, [i], {i}) for i in range(n)]

    for _ in range(n - 1):
        candidates = []
        for cum_score, order, used in beams:
            last = order[-1]
            row = scores[last]
            allowed = confirmed_successors.get(last)
            if edge_mode == "hard":
                next_indices = allowed if allowed else range(n)
            else:
                next_indices = range(n)
            for nxt in next_indices:
                if nxt in used:
                    continue
                if (last, nxt) in impossible:
                    continue
                new_score = cum_score + row[nxt].item()
                if edge_mode == "soft" and allowed and nxt in allowed:
                    new_score += confirmed_bonus
                candidates.append((new_score, order + [nxt], used | {nxt}))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    if beams and len(beams[0][1]) == n:
        if diagnostics is not None:
            diagnostics["fell_back"] = False
            diagnostics["fragments_placed_by_beam"] = n
        return beams[0][1]

    # Fallback: constraints cut off the search before covering every fragment.
    # Extend the best partial beam greedily by raw score, ignoring the
    # impossible set (it has already proven infeasible).
    best = max(beams, key=lambda x: x[0]) if beams else (0.0, [], set())
    order = list(best[1])
    used = set(best[2])
    if diagnostics is not None:
        diagnostics["fell_back"] = True
        diagnostics["fragments_placed_by_beam"] = len(order)
    while len(order) < n:
        remaining = [i for i in range(n) if i not in used]
        if not remaining:
            break
        if order:
            last = order[-1]
            nxt = max(
                remaining,
                key=lambda i: scores[last, i].item()
                + (
                    confirmed_bonus
                    if edge_mode == "soft"
                    and confirmed_successors.get(last)
                    and i in confirmed_successors[last]
                    else 0.0
                ),
            )
        else:
            nxt = remaining[0]
        order.append(nxt)
        used.add(nxt)
    return order
