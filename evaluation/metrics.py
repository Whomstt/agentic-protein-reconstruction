from difflib import SequenceMatcher
from collections import Counter


def sequence_similarity(target, reconstruction):
    """SequenceMatcher ratio — overall structural similarity (0-1)."""
    return SequenceMatcher(None, target, reconstruction).ratio()


def residue_accuracy(target, reconstruction):
    """Position-wise residue match accuracy (0-1)."""
    max_len = max(len(target), len(reconstruction))
    if max_len == 0:
        return 0.0
    correct = sum(a == b for a, b in zip(target.ljust(max_len), reconstruction.ljust(max_len)))
    return correct / max_len


def fragment_accuracy(target, fragments, order):
    """Fraction of fragments placed in the correct position (0-1)."""
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


def cosine_residue_similarity(target, reconstruction):
    """Cosine similarity over residue frequency vectors — composition similarity (0-1)."""
    freq_t = Counter(target)
    freq_r = Counter(reconstruction)
    keys = set(freq_t) | set(freq_r)
    dot = sum(freq_t.get(k, 0) * freq_r.get(k, 0) for k in keys)
    mag_t = sum(v ** 2 for v in freq_t.values()) ** 0.5
    mag_r = sum(v ** 2 for v in freq_r.values()) ** 0.5
    if mag_t == 0 or mag_r == 0:
        return 0.0
    return dot / (mag_t * mag_r)


METRIC_NAMES = {
    "exact_match": "Exact Match",
    "similarity": "Sequence Similarity",
    "residue_acc": "Residue Accuracy",
    "fragment_acc": "Fragment Accuracy",
    "edit_distance": "Edit Distance (Levenshtein)",
    "norm_edit_distance": "Norm. Edit Distance (lower=better)",
    "lcs_ratio": "LCS Ratio",
    "cosine": "Cosine Residue Similarity",
}

# Metrics where raw integer display is more informative
_INTEGER_METRICS = {"edit_distance"}


def compute_all(target, reconstruction, fragments=None, order=None):
    """Compute all metrics. Returns dict keyed by metric name."""
    return {
        "exact_match": exact_match(target, reconstruction),
        "similarity": sequence_similarity(target, reconstruction),
        "residue_acc": residue_accuracy(target, reconstruction),
        "fragment_acc": fragment_accuracy(target, fragments, order) if order else 0.0,
        "edit_distance": levenshtein_distance(target, reconstruction),
        "norm_edit_distance": normalized_levenshtein(target, reconstruction),
        "lcs_ratio": lcs_ratio(target, reconstruction),
        "cosine": cosine_residue_similarity(target, reconstruction),
    }


def print_metrics(metrics):
    """Print a single sample's metrics."""
    for key, label in METRIC_NAMES.items():
        val = metrics[key]
        fmt = f"{val:.0f}" if key in _INTEGER_METRICS else f"{val:.4f}"
        print(f"  {label}: {fmt}")


def print_averages(summary, n):
    """Print averaged metrics across samples."""
    for key, label in METRIC_NAMES.items():
        avg = sum(summary[key]) / n
        fmt = f"{avg:.1f}" if key in _INTEGER_METRICS else f"{avg:.4f}"
        print(f"  {label}: {fmt}")
