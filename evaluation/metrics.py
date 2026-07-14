import math
from difflib import SequenceMatcher
from collections import Counter, deque


def sequence_similarity(target, reconstruction):
    """SequenceMatcher ratio — the single soft string metric (0-1). Because every
    candidate is a permutation of the same fragment multiset, string composition
    is fixed and this ratio reflects how much of the sequence is in the right
    order. Kept as the one string-level view; exact_match is its binary floor."""
    return SequenceMatcher(None, target, reconstruction).ratio()


def exact_match(target, reconstruction):
    """Binary — 1 if perfect reconstruction, 0 otherwise."""
    return 1.0 if target == reconstruction else 0.0


def recover_true_order(target, fragments):
    """Greedy left-to-right tiling: returns the permutation of fragment indices
    that reconstructs the target. Prefers the longest match first to stay robust
    when one fragment is a prefix of another. Returns None if the fragments do
    not tile the target exactly (in which case the ordering metrics that depend
    on a ground-truth order are reported as NaN, not silently as 0)."""
    if not fragments:
        return None
    remaining = sorted(range(len(fragments)), key=lambda i: -len(fragments[i]))
    order = []
    cursor = 0
    while cursor < len(target) and remaining:
        for idx in remaining:
            frag = fragments[idx]
            if target[cursor : cursor + len(frag)] == frag:
                order.append(idx)
                cursor += len(frag)
                remaining.remove(idx)
                break
        else:
            return None
    return order if cursor == len(target) else None


def is_clean_permutation(order, num_fragments):
    """True iff `order` places each fragment index exactly once — i.e. the
    prediction is a genuine permutation of the input set with no dropped,
    duplicated, or out-of-range fragment."""
    if not order or num_fragments <= 0:
        return False
    return sorted(order) == list(range(num_fragments))


def adjacent_pair_accuracy(pred_order, true_order, fragments):
    """Fraction of true adjacent fragment pairs preserved in the prediction (0-1).
    Uses fragment strings (not indices) as a multiset, so duplicate and
    substring-identical fragments are handled correctly (an identical string in a
    different index slot is not spuriously counted as misordered). Directed, so a
    reversed ordering scores 0."""
    if not pred_order or not true_order or len(true_order) < 2:
        return 0.0

    def pair_counter(order):
        return Counter(
            (fragments[order[i]], fragments[order[i + 1]])
            for i in range(len(order) - 1)
        )

    true_pairs = pair_counter(true_order)
    pred_pairs = pair_counter(pred_order)
    common = sum((true_pairs & pred_pairs).values())
    total = sum(true_pairs.values())
    return common / total if total else 0.0


def longest_correct_run(pred_order, true_order, fragments):
    """Length of the longest contiguous run of fragments that appears in the same
    order in both the prediction and the truth, normalized by the number of
    fragments (0-1). Credits a partly-correct assembly: distinguishes "assembled
    one long correct block" from "got scattered adjacencies right". Compared on
    fragment strings (a multiset view), so duplicate fragments do not distort it."""
    if not pred_order or not true_order:
        return 0.0
    pred_seq = [fragments[i] for i in pred_order]
    true_seq = [fragments[i] for i in true_order]
    m, n = len(pred_seq), len(true_seq)
    # Longest common contiguous substring over the fragment-token sequences.
    prev = [0] * (n + 1)
    best = 0
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if pred_seq[i] == true_seq[j]:
                curr[j + 1] = prev[j] + 1
                if curr[j + 1] > best:
                    best = curr[j + 1]
        prev = curr
    return best / len(true_seq)


def _matched_rank_sequence(pred_order, true_order, fragments):
    """Map the predicted order to the sequence of true-order rank positions,
    matching duplicate fragment strings by occurrence (k-th occurrence in the
    prediction -> k-th occurrence in the truth). Fragments not present in the
    truth (extras) are dropped. This makes rank correlation robust to
    duplicate/substring-identical fragments, which a raw index mapping is not."""
    true_seq = [fragments[i] for i in true_order]
    pool = {}
    for pos, s in enumerate(true_seq):
        pool.setdefault(s, deque()).append(pos)
    ranks = []
    for i in pred_order:
        s = fragments[i]
        q = pool.get(s)
        if q:
            ranks.append(q.popleft())
    return ranks


def kendall_tau(pred_order, true_order, fragments):
    """Kendall tau rank correlation between predicted and true fragment orders
    (-1 to 1); 0 is the expected value for a random permutation, 1 a perfect
    match, -1 an exact reversal. Ranks are matched on fragment strings (see
    _matched_rank_sequence) so identical strings in different slots are not
    treated as misordered."""
    if not pred_order or not true_order or len(true_order) < 2:
        return 0.0
    ranks = _matched_rank_sequence(pred_order, true_order, fragments)
    n = len(ranks)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[i] < ranks[j]:
                concordant += 1
            elif ranks[i] > ranks[j]:
                discordant += 1
    total = n * (n - 1) // 2
    return (concordant - discordant) / total if total else 0.0


def junction_ranking_stats(score_matrix, true_order, num_fragments):
    """Search-independent quality of the pLM junction scorer: does it rank each
    true successor at the top, before any search or constraint is applied?

    For every true adjacency i->s in `true_order`, rank all candidate successors
    j != i by score_matrix[i][j] (higher = more plausible). Returns top-1/top-3
    accuracy and mean reciprocal rank over those true junctions. This is the one
    measurement of the pipeline's core assumption; every other metric is measured
    *after* search and entangles scorer quality with search dynamics."""
    if not true_order or len(true_order) < 2 or num_fragments < 2:
        return {"top1_acc": None, "top3_acc": None, "mrr": None, "num_junctions": 0}

    def score(i, j):
        return float(score_matrix[i][j])

    top1 = top3 = 0
    rr_sum = 0.0
    counted = 0
    for k in range(len(true_order) - 1):
        i, s = true_order[k], true_order[k + 1]
        if i >= num_fragments or s >= num_fragments:
            continue
        s_score = score(i, s)
        better = sum(
            1 for j in range(num_fragments) if j != i and j != s and score(i, j) > s_score
        )
        rank = better + 1
        if rank == 1:
            top1 += 1
        if rank <= 3:
            top3 += 1
        rr_sum += 1.0 / rank
        counted += 1

    if counted == 0:
        return {"top1_acc": None, "top3_acc": None, "mrr": None, "num_junctions": 0}
    return {
        "top1_acc": top1 / counted,
        "top3_acc": top3 / counted,
        "mrr": rr_sum / counted,
        "num_junctions": counted,
    }


def rank_concordance(pairs):
    """Given (validity_score, quality) pairs — validity lower-is-better, quality
    higher-is-better — return (concordance, num_comparable): the fraction of
    comparable pairs in which the lower-validity item is also the higher-quality
    one. Ties on either axis are skipped. This is the trust check for the
    selection signal: 0.5 means validity is no better than a coin flip at picking
    the better candidate. Returns (None, 0) when nothing is comparable."""
    pts = [
        (v, q)
        for v, q in pairs
        if isinstance(v, (int, float))
        and isinstance(q, (int, float))
        and not math.isnan(v)
        and not math.isnan(q)
        and not math.isinf(v)
    ]
    concordant = 0
    total = 0
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            v1, q1 = pts[a]
            v2, q2 = pts[b]
            if v1 == v2 or q1 == q2:
                continue
            total += 1
            if (v1 < v2 and q1 > q2) or (v2 < v1 and q2 > q1):
                concordant += 1
    return (concordant / total if total else None, total)


def nanmean(values):
    """Mean over numeric values, skipping None/NaN. Returns NaN if nothing usable
    remains, so downstream formatting shows 'nan' rather than crashing."""
    vals = [
        v for v in values if isinstance(v, (int, float)) and not math.isnan(v)
    ]
    return sum(vals) / len(vals) if vals else float("nan")


# Ordering metrics that require a recovered ground-truth order; NaN for a sample
# whose fragments do not tile the target (recover_true_order returned None).
ORDERING_METRICS = {"adjacent_pair_acc", "longest_correct_run", "kendall_tau"}

METRIC_NAMES = {
    "exact_match": "Exact Match",
    "similarity": "Sequence Similarity",
    "adjacent_pair_acc": "Adjacent Pair Accuracy",
    "longest_correct_run": "Longest Correct Run",
    "kendall_tau": "Kendall Tau",
}

# Metrics where lower values indicate better reconstructions. Empty for the
# current set (all higher-is-better); kept as the single source of truth so
# report/console formatting never hard-codes a metric name.
LOWER_IS_BETTER: set[str] = set()


def compute_all(target, reconstruction, fragments=None, order=None):
    """Compute all metrics. Returns a dict keyed by metric name, plus the
    auxiliary key ``true_order_recovered`` (outside METRIC_NAMES) recording
    whether the ground-truth order could be tiled from the fragments. When it
    could not, the ordering metrics are NaN rather than a misleading 0."""
    true_order = recover_true_order(target, fragments) if fragments else None
    recovered = true_order is not None

    if order and true_order:
        adjacent = adjacent_pair_accuracy(order, true_order, fragments)
        longest = longest_correct_run(order, true_order, fragments)
        tau = kendall_tau(order, true_order, fragments)
    elif order and not recovered:
        adjacent = longest = tau = float("nan")
    else:
        adjacent = longest = tau = 0.0

    return {
        "exact_match": exact_match(target, reconstruction),
        "similarity": sequence_similarity(target, reconstruction),
        "adjacent_pair_acc": adjacent,
        "longest_correct_run": longest,
        "kendall_tau": tau,
        "true_order_recovered": recovered,
    }


def print_comparison(baseline_summary, recon_summary, n):
    """Print averaged metrics side-by-side: shuffled baseline vs reconstructed vs
    delta. Delta is raw (reconstructed - baseline); a trailing tag marks the
    direction of improvement using LOWER_IS_BETTER. NaN-safe."""
    label_width = max(len(label) for label in METRIC_NAMES.values())
    col_base = 10
    col_recon = 14
    col_delta = 18
    header = (
        f"  {'Metric'.ljust(label_width)}  "
        f"{'Shuffled'.rjust(col_base)}  "
        f"{'Reconstructed'.rjust(col_recon)}  "
        f"{'Delta'.rjust(col_delta)}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, label in METRIC_NAMES.items():
        base = nanmean(baseline_summary[key])
        recon = nanmean(recon_summary[key])
        delta = recon - base
        improved = (delta < 0) if key in LOWER_IS_BETTER else (delta > 0)
        tag = "(better)" if improved and delta != 0 else ("(worse)" if delta != 0 else "(same)  ")
        sign = "+" if delta >= 0 else "-"
        delta_str = f"{sign}{abs(delta):.4f} {tag}"
        print(
            f"  {label.ljust(label_width)}  "
            f"{base:>{col_base}.4f}  "
            f"{recon:>{col_recon}.4f}  "
            f"{delta_str:>{col_delta}}"
        )
