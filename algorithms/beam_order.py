from config import cfg


def beam_order(scores, impossible_junctions=None, start_candidates=None):
    n = scores.shape[0]
    beam_size = cfg["mlm_model"]["beam_size"]
    impossible = impossible_junctions or set()

    if start_candidates:
        beams = [(0.0, [i], {i}) for i in start_candidates]
    else:
        beams = [(0.0, [i], {i}) for i in range(n)]

    for _ in range(n - 1):
        candidates = []
        for cum_score, order, used in beams:
            last = order[-1]
            row = scores[last]
            for nxt in range(n):
                if nxt in used:
                    continue
                if (last, nxt) in impossible:
                    continue
                new_score = cum_score + row[nxt].item()
                candidates.append((new_score, order + [nxt], used | {nxt}))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

    if beams and len(beams[0][1]) == n:
        return beams[0][1]

    # Fallback: constraints cut off the search before covering every fragment.
    # Extend the best partial beam greedily by raw score, ignoring the
    # impossible set (it has already proven infeasible).
    best = max(beams, key=lambda x: x[0]) if beams else (0.0, [], set())
    order = list(best[1])
    used = set(best[2])
    while len(order) < n:
        remaining = [i for i in range(n) if i not in used]
        if not remaining:
            break
        if order:
            last = order[-1]
            nxt = max(remaining, key=lambda i: scores[last, i].item())
        else:
            nxt = remaining[0]
        order.append(nxt)
        used.add(nxt)
    return order
