def greedy_order(scores):
    """Greedily pick the next fragment with highest score from current fragment."""
    n = scores.shape[0]
    used = set()
    start = (
        scores.sum(dim=0).argmin().item()
    )  # start with fragment least likely to be after any other
    order = [start]
    used.add(start)
    while len(order) < n:
        row = scores[
            order[-1]
        ].clone()  # consider all candidates after the last fragment in order
        row[list(used)] = -1  # ignore already used fragments
        nxt = row.argmax().item()  # pick highest probability
        order.append(nxt)
        used.add(nxt)
    return order
