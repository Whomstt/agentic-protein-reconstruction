import math
from difflib import SequenceMatcher
from collections import Counter, deque


def sequence_similarity(target, reconstruction):
    """SequenceMatcher ratio (0-1). Composition is identical across candidates,
    so this reflects only how much of the sequence is in the right order."""
    return SequenceMatcher(None, target, reconstruction).ratio()


def exact_match(target, reconstruction):
    """Binary — 1 if perfect reconstruction, 0 otherwise."""
    return 1.0 if target == reconstruction else 0.0


def _levenshtein_py(a, b):
    """Two-row DP. Correct for any input; O(len(a)*len(b)) in pure Python."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _levenshtein_np(a, b, np):
    """Same recurrence, one numpy row per character of `a`.

    The insertion term ``curr[j-1] + 1`` is a left-to-right dependency, so it
    cannot be written as a single vector op. Substituting ``d[j] = curr[j] - j``
    turns it into ``d[j] = min(d[j], d[j-1])`` — a prefix minimum — which
    ``np.minimum.accumulate`` computes exactly. Protein sequences run to a few
    thousand residues, where the pure-Python loop costs seconds per call.
    """
    row = np.frombuffer(b.encode("latin-1"), dtype=np.uint8).astype(np.int32)
    idx = np.arange(len(b) + 1, dtype=np.int32)
    prev = idx.copy()
    curr = np.empty(len(b) + 1, dtype=np.int32)
    for i, ca in enumerate(a, 1):
        curr[0] = i
        np.minimum(prev[1:] + 1, prev[:-1] + (row != ord(ca)), out=curr[1:])
        curr -= idx
        np.minimum.accumulate(curr, out=curr)
        curr += idx
        prev, curr = curr, prev
    return int(prev[-1])


def levenshtein(a, b):
    """Unit-cost edit distance (insert/delete/substitute)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) * len(b) > 4096:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            return _levenshtein_np(a, b, np)
    return _levenshtein_py(a, b)


def edit_similarity(target, reconstruction):
    """1 - (Levenshtein distance / length), i.e. the fraction of residues that do
    not need editing (0-1, higher is better). Reported this way rather than as a
    raw distance so it shares the direction of every other metric.

    Unlike `sequence_similarity`, which uses difflib's recursive
    longest-matching-block heuristic, this is the standard edit distance, so it
    is well defined rather than an artifact of one implementation. Candidates are
    permutations of the same residue multiset, so both strings have equal length
    and the normaliser is just the protein length."""
    denom = max(len(target), len(reconstruction))
    if not denom:
        return 1.0
    return 1.0 - levenshtein(target, reconstruction) / denom


def recover_true_order(target, fragments):
    """Greedy left-to-right tiling of the target, longest fragment first so a
    fragment that prefixes another does not derail it. Returns the permutation of
    fragment indices, or None when the fragments do not tile the target exactly
    (the ordering metrics are then NaN rather than 0)."""
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
    """True iff `order` places each fragment index exactly once: no dropped,
    duplicated or out-of-range fragment."""
    if not order or num_fragments <= 0:
        return False
    return sorted(order) == list(range(num_fragments))


def adjacent_pair_accuracy(pred_order, true_order, fragments):
    """Fraction of true adjacent fragment pairs preserved (0-1). Compares
    fragment strings as a multiset, so duplicate fragments are not read as
    misordered. Directed: a reversed ordering scores 0."""
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
    """Longest contiguous correctly-ordered block over the fragment count (0-1).
    Credits partial assembly, which adjacent_pair_accuracy cannot distinguish from
    scattered correct adjacencies. Compared on fragment strings."""
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
    """Predicted order mapped to true-order rank positions, matching duplicate
    fragment strings by occurrence (k-th predicted -> k-th true) and dropping
    extras. A raw index mapping would misrank duplicates."""
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
    """Kendall tau between predicted and true fragment order (-1 to 1): 0 for a
    random permutation, 1 for a perfect match, -1 for an exact reversal. Ranks are
    matched on fragment strings, see _matched_rank_sequence."""
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


def junction_ranking_stats(score_matrix, true_order, num_fragments, skip_pairs=None):
    """Search-independent quality of the pLM junction scorer: for every true
    adjacency i->s, rank the candidate successors by score_matrix[i][j] and report
    top-1/top-3 accuracy and MRR. Every other metric is measured after search, and
    so entangles scorer quality with search dynamics.

    `skip_pairs` excludes true junctions from the measurement. Pass the overlap
    graph's confirmed junctions when the matrix carries their sentinel values,
    otherwise the result is inflated by adjacencies handed over as known."""
    if not true_order or len(true_order) < 2 or num_fragments < 2:
        return {"top1_acc": None, "top3_acc": None, "mrr": None, "num_junctions": 0}

    skip = {tuple(pair) for pair in skip_pairs} if skip_pairs else set()

    def score(i, j):
        return float(score_matrix[i][j])

    top1 = top3 = 0
    rr_sum = 0.0
    counted = 0
    for k in range(len(true_order) - 1):
        i, s = true_order[k], true_order[k + 1]
        if i >= num_fragments or s >= num_fragments:
            continue
        if (i, s) in skip:
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
    """Trust check for the selection signal. Over (validity, quality) pairs
    (validity lower-is-better), returns the fraction of comparable pairs whose
    lower-validity item is also the higher-quality one, and how many were
    comparable. Ties are skipped; 0.5 is a coin flip."""
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
    """Mean over numeric values, skipping None/NaN. NaN when nothing is usable."""
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
    "edit_similarity": "Edit Similarity",
    "adjacent_pair_acc": "Adjacent Pair Accuracy",
    "longest_correct_run": "Longest Correct Run",
    "kendall_tau": "Kendall Tau",
}

# Metrics where lower is better. Empty for the current set, but kept so report
# and console formatting never hard-code a metric name.
LOWER_IS_BETTER: set[str] = set()


def compute_all(target, reconstruction, fragments=None, order=None):
    """All metrics keyed by name, plus ``true_order_recovered``: false when the
    fragments did not tile the target, in which case the three ordering metrics
    are NaN rather than 0."""
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
        "edit_similarity": edit_similarity(target, reconstruction),
        "adjacent_pair_acc": adjacent,
        "longest_correct_run": longest,
        "kendall_tau": tau,
        "true_order_recovered": recovered,
    }


def print_comparison(baseline_summary, recon_summary, n):
    """Print averaged Random Order vs reconstructed metrics and their raw
    delta, tagged for direction via LOWER_IS_BETTER. NaN-safe."""
    label_width = max(len(label) for label in METRIC_NAMES.values())
    col_base = 12  # wide enough for the "Random Order" heading
    col_recon = 14
    col_delta = 18
    header = (
        f"  {'Metric'.ljust(label_width)}  "
        f"{'Random Order'.rjust(col_base)}  "
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
