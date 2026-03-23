from config import cfg


def beam_order(scores):
    n = scores.shape[0]
    beams = [(0.0, [i], {i}) for i in range(n)]

    for _ in range(n - 1):
        candidates = []
        for cum_score, order, used in beams:
            row = scores[order[-1]]
            for nxt in range(n):
                if nxt in used:
                    continue
                new_score = cum_score + row[nxt].item()
                candidates.append((new_score, order + [nxt], used | {nxt}))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[: cfg["mlm_model"]["beam_size"]]

    return beams[0][1]
