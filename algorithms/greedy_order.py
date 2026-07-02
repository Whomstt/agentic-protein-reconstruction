def greedy_order(
    scores,
    impossible_junctions=None,
    start_candidates=None,
    confirmed_successors=None,
    edge_mode="hard",
    confirmed_bonus=0.0,
):
    """Greedily pick the next fragment with highest score from current fragment."""
    n = scores.shape[0]
    impossible = impossible_junctions or set()
    confirmed_successors = confirmed_successors or {}
    used = set()
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
        nxt = row.argmax().item()  # pick highest probability
        order.append(nxt)
        used.add(nxt)
    return order
