def greedy_order(
    scores,
    impossible_junctions=None,
    start_candidates=None,
    confirmed_successors=None,
    edge_mode="hard",
    confirmed_bonus=0.0,
    diagnostics: dict | None = None,
):
    """Greedily pick the next fragment with highest score from current fragment.

    diagnostics, if passed, is populated in place with forced_impossible_count:
    how many steps had every remaining candidate marked impossible (trypsin
    K/R->P or non-K/R-terminal violations) and had to pick one anyway — a
    sign the junction scores/constraints are fighting each other for this
    fragment set, independent of which levers were chosen.
    """
    n = scores.shape[0]
    impossible = impossible_junctions or set()
    confirmed_successors = confirmed_successors or {}
    used = set()
    forced_impossible_count = 0
    if start_candidates:
        incoming = scores.sum(dim=0)
        start = min(start_candidates, key=lambda i: incoming[i].item())
    else:
        start = (
            scores.sum(dim=0).argmin().item()
        )  # start with fragment least likely to be after any other
    order = [start]
    used.add(start)
    while len(order) < n:
        row = scores[
            order[-1]
        ].clone()  # consider all candidates after the last fragment in order
        row[list(used)] = float("-inf")  # ignore already used fragments
        for idx in range(n):
            if (order[-1], idx) in impossible:
                row[idx] = float("-inf")
                continue
            if edge_mode == "soft":
                allowed = confirmed_successors.get(order[-1])
                if allowed and idx in allowed:
                    row[idx] = row[idx].item() + confirmed_bonus
        if bool((row == float("-inf")).all()):
            forced_impossible_count += 1
            row = scores[order[-1]].clone()
            row[list(used)] = float("-inf")
        nxt = row.argmax().item()  # pick highest probability
        order.append(nxt)
        used.add(nxt)
    if diagnostics is not None:
        diagnostics["forced_impossible_count"] = forced_impossible_count
    return order
