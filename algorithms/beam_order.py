from config import cfg


def beam_order(scores, impossible_junctions=None, start_candidates=None):
    n = scores.shape[0]
    beam_size = cfg["mlm_model"]["beam_size"]
    impossible = impossible_junctions or set()

    if start_candidates:
        beams = [(0.0, [i], {i}) for i in start_candidates]
    else:
        beams = [(0.0, [i], {i}) for i in range(n)]

    for step in range(n - 1):
        candidates = []

        for cum_score, order, used in beams:
            last = order[-1]
            row = scores[last]
            for nxt in range(n):
                if nxt in used:
                    continue
                # Trypsin chemistry: K/R → P and non-K/R outgoing are impossible
                if (last, nxt) in impossible:
                    continue
                new_score = cum_score + row[nxt].item()
                candidates.append((new_score, order + [nxt], used | {nxt}))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

    return beams[0][1]
