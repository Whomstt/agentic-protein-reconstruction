"""Inferential statistics for the evaluation reports.

Pure functions: no I/O, no config, no model loading, no global RNG use. Every
interval and p-value in a generated report comes from here.

stdlib only. scipy is not installed and the rebuild CLI must run without a GPU
or network, so the tests, exact distributions and percentile helper are
implemented here and checked against independent brute-force implementations in
tests/test_stats.py. Randomized routines take an explicit seed and use their own
random.Random, so results do not depend on the global RNG. NaN inputs are
dropped rather than propagated (ordering metrics are NaN when a sample's
fragments do not tile the target) and every result carries the n actually used."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from statistics import NormalDist

_NORM = NormalDist()

# Metrics are reported together, so a report that claims significance on any of
# them is running five tests; Holm correction is applied across this family.
METRIC_FAMILY = (
    "exact_match",
    "similarity",
    "adjacent_pair_acc",
    "longest_correct_run",
    "kendall_tau",
)


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval. ``method`` records how the
    interval was produced, including any fallback, so a report cannot mislabel its
    own maths."""

    point: float
    low: float
    high: float
    n: int
    confidence: float
    method: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TestResult:
    """A hypothesis test outcome. ``detail`` carries the counts needed to judge it:
    discordant pairs for McNemar, non-zero differences for Wilcoxon."""

    name: str
    statistic: float
    pvalue: float
    n: int
    detail: dict

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def clean(values) -> list[float]:
    """Drop None/NaN/inf, returning plain floats, so a NaN ordering metric shrinks n
    rather than poisoning the statistic."""
    out = []
    for v in values:
        if _is_number(v) and not math.isnan(float(v)) and not math.isinf(float(v)):
            out.append(float(v))
    return out


def clean_pairs(xs, ys) -> tuple[list[float], list[float]]:
    """Drop pairs where either side is missing: a paired test must compare the same
    samples on both arms, so a sample is dropped from both or neither."""
    a, b = [], []
    for x, y in zip(xs, ys):
        if (
            _is_number(x)
            and _is_number(y)
            and not math.isnan(float(x))
            and not math.isnan(float(y))
            and not math.isinf(float(x))
            and not math.isinf(float(y))
        ):
            a.append(float(x))
            b.append(float(y))
    return a, b


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100]) over an already-sorted list,
    matching numpy's default 'linear' method. Implemented here to stay
    dependency-free; the test suite checks it against numpy."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    q = min(max(q, 0.0), 100.0)
    pos = (len(sorted_values) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def mean(values) -> float:
    vals = clean(values)
    return sum(vals) / len(vals) if vals else float("nan")


# --------------------------------------------------------------------------
# Binomial proportion interval
# --------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion, used for Exact Match.

    Wald is wrong at the small proportions this task produces: it undercovers, can
    put the lower bound below zero, and collapses to [0, 0] at zero successes.
    Wilson stays inside [0, 1] and keeps sensible width at the boundaries."""
    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, confidence, "wilson")
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")

    z = _NORM.inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    # At the boundaries the exact bound is 0 (resp. 1); pin it so a report never
    # prints floating-point dust like 2.8e-17 for "no exact matches".
    low = 0.0 if successes == 0 else max(0.0, center - margin)
    high = 1.0 if successes == n else min(1.0, center + margin)
    return Interval(
        point=p,
        low=low,
        high=high,
        n=n,
        confidence=confidence,
        method="wilson",
    )


# --------------------------------------------------------------------------
# BCa bootstrap
# --------------------------------------------------------------------------


def _jackknife_acceleration(values: list[float], statistic) -> float:
    """Acceleration from the jackknife distribution's skewness. Zero when the
    statistic is symmetric across leave-one-out samples."""
    n = len(values)
    thetas = []
    for i in range(n):
        thetas.append(statistic(values[:i] + values[i + 1 :]))
    theta_bar = sum(thetas) / n
    num = sum((theta_bar - t) ** 3 for t in thetas)
    den = sum((theta_bar - t) ** 2 for t in thetas)
    if den <= 0:
        return 0.0
    return num / (6.0 * den**1.5)


def bca_bootstrap_ci(
    values,
    statistic=None,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 20260726,
) -> Interval:
    """Bias-corrected and accelerated (BCa) bootstrap CI for a statistic of one
    sample, defaulting to the mean.

    BCa rather than a plain percentile bootstrap because these metrics are bounded
    and skewed near the floor, where percentile intervals are visibly off-centre.
    BCa corrects for median bias (z0) and for the statistic's variance changing with
    its value (a). Resampling uses a private random.Random(seed), so the same input
    and seed always give the same interval.

    Fallbacks, recorded in ``Interval.method`` rather than applied silently: fewer
    than 2 usable values gives a NaN interval; a constant sample gives a degenerate
    interval at the point; bootstrap replicates all on one side of the observed
    statistic make z0 infinite and fall back to a percentile interval."""
    statistic = statistic or (lambda vs: sum(vs) / len(vs))
    vals = clean(values)
    n = len(vals)
    if n < 2:
        point = statistic(vals) if vals else float("nan")
        return Interval(point, float("nan"), float("nan"), n, confidence, "bca (insufficient n)")

    theta_hat = statistic(vals)
    if all(v == vals[0] for v in vals):
        return Interval(theta_hat, theta_hat, theta_hat, n, confidence, "bca (degenerate: constant sample)")

    rng = random.Random(seed)
    replicates = []
    for _ in range(n_resamples):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        replicates.append(statistic(sample))
    replicates.sort()

    alpha = 1.0 - confidence
    n_below = sum(1 for r in replicates if r < theta_hat)
    prop = n_below / n_resamples
    if prop <= 0.0 or prop >= 1.0:
        return Interval(
            point=theta_hat,
            low=percentile(replicates, 100.0 * alpha / 2.0),
            high=percentile(replicates, 100.0 * (1.0 - alpha / 2.0)),
            n=n,
            confidence=confidence,
            method="percentile (BCa z0 undefined)",
        )

    z0 = _NORM.inv_cdf(prop)
    a = _jackknife_acceleration(vals, statistic)

    def adjusted(z_alpha: float) -> float:
        num = z0 + z_alpha
        denom = 1.0 - a * num
        if denom <= 0:
            return float("nan")
        return _NORM.cdf(z0 + num / denom)

    lo_q = adjusted(_NORM.inv_cdf(alpha / 2.0))
    hi_q = adjusted(_NORM.inv_cdf(1.0 - alpha / 2.0))
    if math.isnan(lo_q) or math.isnan(hi_q):
        return Interval(
            point=theta_hat,
            low=percentile(replicates, 100.0 * alpha / 2.0),
            high=percentile(replicates, 100.0 * (1.0 - alpha / 2.0)),
            n=n,
            confidence=confidence,
            method="percentile (BCa acceleration unstable)",
        )

    return Interval(
        point=theta_hat,
        low=percentile(replicates, 100.0 * lo_q),
        high=percentile(replicates, 100.0 * hi_q),
        n=n,
        confidence=confidence,
        method="bca",
    )


def bca_paired_delta_ci(
    arm_a,
    arm_b,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 20260726,
) -> Interval:
    """BCa CI for the mean paired difference (arm_a - arm_b), resampling whole
    samples so the pairing is preserved. The arms run on the same proteins, so an
    interval built from their marginal CIs would be wrong."""
    a, b = clean_pairs(arm_a, arm_b)
    deltas = [x - y for x, y in zip(a, b)]
    return bca_bootstrap_ci(
        deltas, confidence=confidence, n_resamples=n_resamples, seed=seed
    )


# --------------------------------------------------------------------------
# Paired tests
# --------------------------------------------------------------------------


def _binom_two_sided_p(k: int, n: int) -> float:
    """Two-sided exact binomial p under p=0.5, as used by exact McNemar:
    2 * P(X <= min(k, n-k)), capped at 1."""
    if n <= 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def mcnemar_exact(arm_a, arm_b, name: str = "mcnemar") -> TestResult:
    """Exact McNemar test on a paired binary outcome, used for Exact Match.

    The arms are paired on the same protein, so only the discordant samples carry
    information: how many proteins A solved that B did not (n10) and vice versa
    (n01). The exact binomial form is used rather than chi-square because the
    discordant counts here are single digits. Inputs are truthy/0-1 per sample."""
    a, b = clean_pairs(arm_a, arm_b)
    n01 = sum(1 for x, y in zip(a, b) if x <= 0 and y > 0)  # only B succeeded
    n10 = sum(1 for x, y in zip(a, b) if x > 0 and y <= 0)  # only A succeeded
    both = sum(1 for x, y in zip(a, b) if x > 0 and y > 0)
    neither = sum(1 for x, y in zip(a, b) if x <= 0 and y <= 0)
    discordant = n01 + n10
    p = _binom_two_sided_p(n10, discordant) if discordant else 1.0
    return TestResult(
        name=name,
        statistic=float(n10),
        pvalue=p,
        n=len(a),
        detail={
            "n10_only_a": n10,
            "n01_only_b": n01,
            "discordant": discordant,
            "both": both,
            "neither": neither,
            "method": "exact binomial (p=0.5)",
        },
    )


def _wilcoxon_exact_p(w: float, ranks: list[int]) -> float:
    """Exact two-sided p for the signed-rank statistic by subset-sum DP over the
    integer ranks: how many of the 2^n sign assignments are at least as extreme as
    observed."""
    n = len(ranks)
    total = sum(ranks)
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in ranks:
        for s in range(total, r - 1, -1):
            if counts[s - r]:
                counts[s] += counts[s - r]
    space = 2.0**n
    w_min = min(w, total - w)
    at_or_below = sum(counts[s] for s in range(0, int(math.floor(w_min)) + 1))
    return min(1.0, 2.0 * at_or_below / space)


def wilcoxon_signed_rank(arm_a, arm_b, name: str = "wilcoxon") -> TestResult:
    """Paired Wilcoxon signed-rank test on arm_a - arm_b.

    The per-sample gains are bounded, discrete-ish and not normal, and the arms are
    paired on the same protein. Zero differences are dropped, Wilcoxon's standard
    handling, which matters here because many proteins give the two arms an
    identical ordering.

    Exact p by subset-sum enumeration when the absolute differences have no ties and
    n is small enough; otherwise the normal approximation with tie and continuity
    correction. The branch taken is recorded in ``detail['method']``."""
    a, b = clean_pairs(arm_a, arm_b)
    diffs = [x - y for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return TestResult(
            name=name,
            statistic=float("nan"),
            pvalue=1.0,
            n=0,
            detail={
                "n_pairs": len(diffs),
                "n_nonzero": 0,
                "n_zero": len(diffs),
                "n_positive": 0,
                "n_negative": 0,
                "method": "no non-zero differences",
            },
        )

    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    tie_groups: list[int] = []
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        tie_groups.append(j - i + 1)
        i = j + 1

    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    w = min(w_plus, w_minus)
    n_pos = sum(1 for d in nonzero if d > 0)
    n_neg = sum(1 for d in nonzero if d < 0)

    has_ties = any(g > 1 for g in tie_groups)
    if not has_ties and n <= 50:
        p = _wilcoxon_exact_p(w, list(range(1, n + 1)))
        method = "exact (subset-sum enumeration)"
    else:
        mu = n * (n + 1) / 4.0
        tie_term = sum(g**3 - g for g in tie_groups)
        var = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
        if var <= 0:
            p = 1.0
            method = "normal approximation (zero variance)"
        else:
            z = (abs(w - mu) - 0.5) / math.sqrt(var)
            p = min(1.0, 2.0 * (1.0 - _NORM.cdf(max(z, 0.0))))
            method = "normal approximation (tie + continuity corrected)"

    return TestResult(
        name=name,
        statistic=float(w),
        pvalue=p,
        n=n,
        detail={
            "n_pairs": len(diffs),
            "n_nonzero": n,
            "n_zero": len(diffs) - n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "w_plus": w_plus,
            "w_minus": w_minus,
            "median_delta": sorted(diffs)[len(diffs) // 2] if diffs else float("nan"),
            "method": method,
        },
    )


# --------------------------------------------------------------------------
# Multiple-comparison correction
# --------------------------------------------------------------------------


def holm_bonferroni(pvalues: dict | list, alpha: float = 0.05):
    """Holm step-down correction across a family of tests.

    Five metrics are tested on the same pair of arms, so an uncorrected 0.05
    threshold would give roughly a 1-in-4 chance of a spurious result. Holm controls
    the family-wise error rate and is uniformly more powerful than Bonferroni.

    Accepts a dict (name -> p) or a list and returns the same shape. Adjusted values
    are enforced monotone non-decreasing in p, which makes 'reject iff adjusted
    p <= alpha' equivalent to the step-down procedure, and capped at 1.0."""
    is_dict = isinstance(pvalues, dict)
    items = list(pvalues.items()) if is_dict else list(enumerate(pvalues))
    usable = [(k, float(p)) for k, p in items if _is_number(p) and not math.isnan(float(p))]
    m = len(usable)
    if m == 0:
        empty: dict = {}
        return empty if is_dict else []

    ordered = sorted(usable, key=lambda kv: kv[1])
    adjusted: dict = {}
    running = 0.0
    for i, (key, p) in enumerate(ordered):
        value = (m - i) * p
        running = max(running, value)  # enforce monotonicity
        adjusted[key] = min(1.0, running)

    if is_dict:
        return {
            key: {
                "p_raw": dict(usable)[key],
                "p_adjusted": adjusted[key],
                "reject": adjusted[key] <= alpha,
            }
            for key, _ in items
            if key in adjusted
        }
    out = [None] * len(items)
    for key, _ in items:
        if key in adjusted:
            out[key] = adjusted[key]
    return out


def compare_arms(
    arm_a: dict,
    arm_b: dict,
    binary_metrics=("exact_match",),
    metrics=METRIC_FAMILY,
    alpha: float = 0.05,
) -> dict:
    """Full paired comparison of two arms across the metric family.

    ``arm_a``/``arm_b`` map metric name -> per-sample values, aligned by sample.
    Exact Match goes through exact McNemar, the four continuous metrics through
    Wilcoxon signed-rank, and Holm is applied across all five together. Returns the
    per-metric test results with the paired delta CI attached, plus the Holm table."""
    results = {}
    raw_p = {}
    for metric in metrics:
        a = arm_a.get(metric, [])
        b = arm_b.get(metric, [])
        if metric in binary_metrics:
            test = mcnemar_exact(a, b, name=f"{metric}:mcnemar_exact")
        else:
            test = wilcoxon_signed_rank(a, b, name=f"{metric}:wilcoxon")
        results[metric] = {
            "test": test.as_dict(),
            "delta_ci": bca_paired_delta_ci(a, b).as_dict(),
        }
        raw_p[metric] = test.pvalue

    holm = holm_bonferroni(raw_p, alpha=alpha)
    for metric, entry in holm.items():
        results[metric]["holm"] = entry
    return {"metrics": results, "alpha": alpha, "family_size": len(raw_p)}
