from config import cfg


def beam_order(scores, start_candidates=None, terminal_candidates=None):
    n = scores.shape[0]
    beam_size = cfg["mlm_model"]["beam_size"]
    terminal_set = set(terminal_candidates) if terminal_candidates else set()

    if start_candidates:
        beams = [(0.0, [i], {i}) for i in start_candidates]
    else:
        beams = [(0.0, [i], {i}) for i in range(n)]

    for step in range(n - 1):
        candidates = []
        is_last_step = step == n - 2

        for cum_score, order, used in beams:
            row = scores[order[-1]]
            for nxt in range(n):
                if nxt in used:
                    continue
                # Trypsin constraints: terminal-only fragments cannot appear in internal positions
                if not is_last_step and nxt in terminal_set:
                    continue
                # On last step, only terminal candidates are valid (if any identified)
                if is_last_step and terminal_set and nxt not in terminal_set:
                    continue
                new_score = cum_score + row[nxt].item()
                candidates.append((new_score, order + [nxt], used | {nxt}))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

    return beams[0][1]
