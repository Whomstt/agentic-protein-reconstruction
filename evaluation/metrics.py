from difflib import SequenceMatcher
from collections import Counter


def sequence_similarity(target, reconstruction):
    """SequenceMatcher ratio — overall structural similarity (0-1)."""
    return SequenceMatcher(None, target, reconstruction).ratio()


def fragment_accuracy(target, fragments, order):
    """Fraction of fragments placed at their correct position in the target (0-1)."""
    if not order or not fragments:
        return 0.0
    cursor = 0
    correct = 0
    for idx in order:
        frag = fragments[idx]
        if target[cursor : cursor + len(frag)] == frag:
            correct += 1
        cursor += len(frag)
    return correct / len(fragments)


def levenshtein_distance(a, b):
    """Edit distance — minimum single-character edits to transform a into b."""
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def normalized_levenshtein(target, reconstruction):
    """Levenshtein distance normalized by the longer sequence length (0-1, lower is better)."""
    max_len = max(len(target), len(reconstruction))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(target, reconstruction) / max_len


def longest_common_subsequence(a, b):
    """Length of the longest common subsequence."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev = curr
    return prev[n]


def lcs_ratio(target, reconstruction):
    """LCS length normalized by target length (0-1) — order-preserving coverage."""
    if len(target) == 0:
        return 0.0
    return longest_common_subsequence(target, reconstruction) / len(target)


def exact_match(target, reconstruction):
    """Binary — 1 if perfect reconstruction, 0 otherwise."""
    return 1.0 if target == reconstruction else 0.0


def recover_true_order(target, fragments):
    """Greedy left-to-right tiling: returns the permutation of fragment indices that
    reconstructs the target. Prefers longest match first to stay robust when one
    fragment is a prefix of another. Returns None if fragments don't tile target."""
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


def adjacent_pair_accuracy(pred_order, true_order, fragments):
    """Fraction of true adjacent fragment pairs preserved in the prediction (0-1).
    Uses fragment strings (not indices) so repeated fragments are handled as a multiset."""
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


def kendall_tau(pred_order, true_order):
    """Kendall tau rank correlation between predicted and true fragment orders (-1 to 1).
    0 is the expected value for a random permutation; 1 is a perfect match."""
    if not pred_order or not true_order or len(true_order) < 2:
        return 0.0
    true_pos = {idx: p for p, idx in enumerate(true_order)}
    ranks = [true_pos[idx] for idx in pred_order if idx in true_pos]
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
    return (concordant - discordant) / total


METRIC_NAMES = {
    "exact_match": "Exact Match",
    "similarity": "Sequence Similarity",
    "fragment_acc": "Fragment Accuracy",
    "norm_edit_distance": "Norm. Edit Dist. (lower=better)",
    "lcs_ratio": "LCS Ratio",
    "adjacent_pair_acc": "Adjacent Pair Accuracy",
    "kendall_tau": "Kendall Tau",
}

# Metrics where lower values indicate better reconstructions.
LOWER_IS_BETTER = {"norm_edit_distance"}


def compute_all(target, reconstruction, fragments=None, order=None):
    """Compute all metrics. Returns dict keyed by metric name."""
    true_order = recover_true_order(target, fragments) if fragments else None
    return {
        "exact_match": exact_match(target, reconstruction),
        "similarity": sequence_similarity(target, reconstruction),
        "fragment_acc": fragment_accuracy(target, fragments, order) if order else 0.0,
        "norm_edit_distance": normalized_levenshtein(target, reconstruction),
        "lcs_ratio": lcs_ratio(target, reconstruction),
        "adjacent_pair_acc": (
            adjacent_pair_accuracy(order, true_order, fragments)
            if order and true_order
            else 0.0
        ),
        "kendall_tau": (
            kendall_tau(order, true_order) if order and true_order else 0.0
        ),
    }


def print_metrics(metrics):
    """Print a single sample's metrics."""
    for key, label in METRIC_NAMES.items():
        print(f"  {label}: {metrics[key]:.4f}")


def print_averages(summary, n):
    """Print averaged metrics across samples."""
    for key, label in METRIC_NAMES.items():
        avg = sum(summary[key]) / n
        print(f"  {label}: {avg:.4f}")


def print_comparison(baseline_summary, recon_summary, n):
    """Print averaged metrics side-by-side: shuffled baseline vs reconstructed vs delta.
    Delta is raw (reconstructed - baseline); a trailing tag marks direction of improvement,
    since some metrics are lower-is-better."""
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
        base = sum(baseline_summary[key]) / n
        recon = sum(recon_summary[key]) / n
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
