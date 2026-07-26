"""Unit tests for evaluation/stats.py.

Run:  python tests/test_stats.py          (or: python -m unittest tests.test_stats)

pytest is not installed in this project, so these use stdlib unittest.

Where possible the tests check against something INDEPENDENT of the
implementation under test — brute-force enumeration of the full sign space for
Wilcoxon, direct binomial summation for McNemar, numpy for the percentile
helper, and published textbook values for Wilson — rather than re-deriving the
same formula twice and calling that agreement.
"""

from __future__ import annotations

import itertools
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.stats import (  # noqa: E402
    Interval,
    bca_bootstrap_ci,
    bca_paired_delta_ci,
    clean,
    clean_pairs,
    compare_arms,
    holm_bonferroni,
    mcnemar_exact,
    percentile,
    wilcoxon_signed_rank,
    wilson_interval,
)


class TestHelpers(unittest.TestCase):
    def test_clean_drops_nan_none_inf_and_bools(self):
        self.assertEqual(
            clean([1.0, None, float("nan"), 2.0, float("inf"), True, 3]),
            [1.0, 2.0, 3.0],
        )

    def test_clean_pairs_drops_pair_when_either_side_missing(self):
        a, b = clean_pairs([1.0, float("nan"), 3.0, 4.0], [1.0, 2.0, None, 8.0])
        self.assertEqual((a, b), ([1.0, 4.0], [1.0, 8.0]))

    def test_percentile_matches_numpy(self):
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is present in this project
            self.skipTest("numpy not installed")
        for data in ([1.0, 2.0, 3.0, 4.0], [5.0], [0.0, 0.0, 1.0], list(range(17))):
            values = sorted(float(v) for v in data)
            for q in (0.0, 2.5, 25.0, 50.0, 97.5, 100.0):
                self.assertAlmostEqual(
                    percentile(values, q), float(np.percentile(values, q)), places=12,
                    msg=f"data={data} q={q}",
                )


class TestWilson(unittest.TestCase):
    def test_textbook_symmetric_case(self):
        """5/10 at 95% is the standard worked example: (0.2366, 0.7634)."""
        ci = wilson_interval(5, 10)
        self.assertAlmostEqual(ci.point, 0.5, places=12)
        self.assertAlmostEqual(ci.low, 0.2365931, places=6)
        self.assertAlmostEqual(ci.high, 0.7634069, places=6)

    def test_zero_successes_is_not_degenerate(self):
        """The reason Wilson is used instead of Wald: at 0 successes Wald gives
        the useless [0, 0], Wilson gives a real upper bound (0/10 -> ~0.2775)."""
        ci = wilson_interval(0, 10)
        self.assertEqual(ci.low, 0.0)
        self.assertAlmostEqual(ci.high, 0.2775328, places=6)

    def test_bounds_stay_inside_unit_interval(self):
        for k in range(0, 101):
            ci = wilson_interval(k, 100)
            self.assertGreaterEqual(ci.low, 0.0)
            self.assertLessEqual(ci.high, 1.0)
            self.assertLessEqual(ci.low, ci.point)
            self.assertGreaterEqual(ci.high, ci.point)

    def test_interval_narrows_as_n_grows(self):
        wide = wilson_interval(10, 100)
        narrow = wilson_interval(100, 1000)
        self.assertLess(narrow.high - narrow.low, wide.high - wide.low)

    def test_higher_confidence_is_wider(self):
        c90 = wilson_interval(11, 100, confidence=0.90)
        c99 = wilson_interval(11, 100, confidence=0.99)
        self.assertLess(c90.high - c90.low, c99.high - c99.low)

    def test_invalid_inputs(self):
        self.assertEqual(wilson_interval(0, 0).n, 0)
        with self.assertRaises(ValueError):
            wilson_interval(11, 10)


class TestBCaBootstrap(unittest.TestCase):
    def test_deterministic_for_fixed_seed(self):
        data = [0.1, 0.4, 0.35, 0.8, 0.2, 0.55, 0.6, 0.05, 0.9, 0.3]
        first = bca_bootstrap_ci(data, n_resamples=2000, seed=42)
        second = bca_bootstrap_ci(data, n_resamples=2000, seed=42)
        self.assertEqual(first, second)

    def test_different_seed_changes_interval_but_not_point(self):
        data = [0.1, 0.4, 0.35, 0.8, 0.2, 0.55, 0.6, 0.05, 0.9, 0.3]
        a = bca_bootstrap_ci(data, n_resamples=2000, seed=1)
        b = bca_bootstrap_ci(data, n_resamples=2000, seed=2)
        self.assertAlmostEqual(a.point, b.point, places=12)
        self.assertNotEqual((a.low, a.high), (b.low, b.high))

    def test_unaffected_by_global_rng_state(self):
        """Uses a private Random instance, so seeding the global RNG elsewhere
        must not move the interval."""
        import random as _random

        data = [0.2, 0.9, 0.1, 0.44, 0.6, 0.31, 0.77, 0.05]
        _random.seed(7)
        a = bca_bootstrap_ci(data, n_resamples=1500, seed=99)
        _random.seed(123456)
        [_random.random() for _ in range(50)]
        b = bca_bootstrap_ci(data, n_resamples=1500, seed=99)
        self.assertEqual(a, b)

    def test_point_estimate_is_the_mean_and_ci_brackets_it(self):
        data = [0.1, 0.4, 0.35, 0.8, 0.2, 0.55, 0.6, 0.05, 0.9, 0.3]
        ci = bca_bootstrap_ci(data, n_resamples=4000, seed=5)
        self.assertAlmostEqual(ci.point, sum(data) / len(data), places=12)
        self.assertLessEqual(ci.low, ci.point)
        self.assertGreaterEqual(ci.high, ci.point)
        self.assertEqual(ci.method, "bca")

    def test_covers_true_mean_for_a_known_distribution(self):
        """Sanity on coverage: a large clean sample's CI must contain the value
        the data was generated around, and be tight."""
        import random as _random

        rng = _random.Random(11)
        data = [rng.gauss(0.5, 0.1) for _ in range(400)]
        ci = bca_bootstrap_ci(data, n_resamples=3000, seed=3)
        self.assertLess(ci.low, 0.5)
        self.assertGreater(ci.high, 0.5)
        self.assertLess(ci.high - ci.low, 0.05)

    def test_constant_sample_is_degenerate_not_a_crash(self):
        ci = bca_bootstrap_ci([0.25] * 20, n_resamples=500, seed=1)
        self.assertEqual((ci.low, ci.point, ci.high), (0.25, 0.25, 0.25))
        self.assertIn("degenerate", ci.method)

    def test_insufficient_n_returns_nan_interval(self):
        ci = bca_bootstrap_ci([0.3], n_resamples=500, seed=1)
        self.assertTrue(math.isnan(ci.low) and math.isnan(ci.high))
        self.assertEqual(ci.n, 1)
        ci_empty = bca_bootstrap_ci([], n_resamples=500, seed=1)
        self.assertEqual(ci_empty.n, 0)

    def test_nan_values_are_dropped_and_n_reflects_it(self):
        ci = bca_bootstrap_ci(
            [0.1, float("nan"), 0.4, 0.35, None, 0.8], n_resamples=1000, seed=1
        )
        self.assertEqual(ci.n, 4)

    def test_kendall_tau_style_negative_values_supported(self):
        data = [-0.4, -0.1, 0.2, -0.6, 0.05, -0.3, 0.1, -0.55]
        ci = bca_bootstrap_ci(data, n_resamples=2000, seed=8)
        self.assertLess(ci.low, 0.0)
        self.assertLessEqual(ci.low, ci.point)
        self.assertGreaterEqual(ci.high, ci.point)

    def test_paired_delta_ci_preserves_pairing(self):
        """A constant +0.1 shift must give a CI tight around 0.1 even though
        each arm alone is highly variable — that is the point of pairing."""
        a = [0.1, 0.9, 0.4, 0.7, 0.2, 0.55, 0.33, 0.81]
        b = [v - 0.1 for v in a]
        ci = bca_paired_delta_ci(a, b, n_resamples=1000, seed=4)
        self.assertAlmostEqual(ci.point, 0.1, places=10)
        self.assertAlmostEqual(ci.low, 0.1, places=8)
        self.assertAlmostEqual(ci.high, 0.1, places=8)


class TestMcNemar(unittest.TestCase):
    @staticmethod
    def _brute_force_two_sided(n10: int, n01: int) -> float:
        """Independent reference: exact two-sided binomial tail under p=0.5,
        summed directly over the discordant space."""
        n = n10 + n01
        if n == 0:
            return 1.0
        probs = [math.comb(n, i) / 2**n for i in range(n + 1)]
        observed = probs[n10]
        return min(1.0, sum(p for p in probs if p <= observed * (1 + 1e-12)))

    def test_matches_independent_binomial_reference(self):
        for n10, n01 in [(3, 12), (0, 5), (1, 1), (7, 2), (10, 10), (0, 0), (2, 9)]:
            a = [1] * n10 + [0] * n01 + [1, 0]
            b = [0] * n10 + [1] * n01 + [1, 0]
            result = mcnemar_exact(a, b)
            self.assertAlmostEqual(
                result.pvalue,
                self._brute_force_two_sided(n10, n01),
                places=12,
                msg=f"n10={n10} n01={n01}",
            )

    def test_known_value(self):
        """n01=3, n10=12 -> 2 * P(X<=3 | n=15, p=0.5) = 1152/32768."""
        a = [1] * 12 + [0] * 3
        b = [0] * 12 + [1] * 3
        self.assertAlmostEqual(mcnemar_exact(a, b).pvalue, 1152 / 32768, places=12)

    def test_reports_discordant_counts(self):
        a = [1, 1, 0, 0, 1, 0]
        b = [0, 1, 1, 0, 1, 0]
        r = mcnemar_exact(a, b)
        self.assertEqual(r.detail["n10_only_a"], 1)
        self.assertEqual(r.detail["n01_only_b"], 1)
        self.assertEqual(r.detail["discordant"], 2)
        self.assertEqual(r.detail["both"], 2)
        self.assertEqual(r.detail["neither"], 2)
        self.assertEqual(r.n, 6)

    def test_concordant_only_gives_p_of_one(self):
        r = mcnemar_exact([1, 1, 0, 0], [1, 1, 0, 0])
        self.assertEqual(r.pvalue, 1.0)
        self.assertEqual(r.detail["discordant"], 0)

    def test_symmetric_in_arms(self):
        a = [1, 1, 1, 0, 0, 0, 1, 0]
        b = [0, 0, 1, 1, 0, 1, 1, 0]
        self.assertAlmostEqual(
            mcnemar_exact(a, b).pvalue, mcnemar_exact(b, a).pvalue, places=12
        )

    def test_float_metric_values_treated_as_binary(self):
        r = mcnemar_exact([1.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        self.assertEqual(r.detail["n10_only_a"], 1)
        self.assertEqual(r.detail["n01_only_b"], 0)


class TestWilcoxon(unittest.TestCase):
    @staticmethod
    def _brute_force_p(diffs: list[float]) -> float:
        """Independent reference: enumerate all 2^n sign flips of the absolute
        differences and count how many give a statistic at least as extreme."""
        nonzero = [d for d in diffs if d != 0]
        n = len(nonzero)
        magnitudes = sorted(abs(d) for d in nonzero)
        ranks = {}
        for i, m in enumerate(magnitudes):
            ranks.setdefault(m, []).append(i + 1)
        rank_of = {m: sum(rs) / len(rs) for m, rs in ranks.items()}
        observed_plus = sum(rank_of[abs(d)] for d in nonzero if d > 0)
        observed = min(observed_plus, sum(rank_of[abs(d)] for d in nonzero) - observed_plus)
        total = sum(rank_of[abs(d)] for d in nonzero)
        extreme = 0
        for signs in itertools.product([1, -1], repeat=n):
            w_plus = sum(rank_of[abs(d)] for d, s in zip(nonzero, signs) if s > 0)
            w = min(w_plus, total - w_plus)
            if w <= observed + 1e-12:
                extreme += 1
        return min(1.0, extreme / 2**n)

    def test_exact_matches_brute_force_enumeration(self):
        cases = [
            [1.0, 2.0, 3.0, 4.0, 5.0, -6.0],
            [0.1, -0.2, 0.3, 0.45, -0.05, 0.6, 0.7],
            [-1.0, -2.0, -3.0, 4.0],
            [0.5, 0.25, 0.125, 0.0625, 0.03125],
        ]
        for diffs in cases:
            a = [d for d in diffs]
            b = [0.0] * len(diffs)
            got = wilcoxon_signed_rank(a, b)
            self.assertAlmostEqual(
                got.pvalue, self._brute_force_p(diffs), places=10, msg=str(diffs)
            )
            self.assertIn("exact", got.detail["method"])

    def test_known_small_case(self):
        """d = [1,2,3,4,5,-6]: W- = 6, and 14 of the 64 sign assignments reach
        a statistic <= 6, so two-sided p = 2*14/64 = 0.4375."""
        r = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0, -6.0], [0.0] * 6)
        self.assertAlmostEqual(r.statistic, 6.0, places=12)
        self.assertAlmostEqual(r.pvalue, 0.4375, places=12)

    def test_zero_differences_are_dropped(self):
        a = [0.5, 0.5, 0.5, 0.9, 0.1]
        b = [0.5, 0.5, 0.5, 0.2, 0.4]
        r = wilcoxon_signed_rank(a, b)
        self.assertEqual(r.detail["n_zero"], 3)
        self.assertEqual(r.detail["n_nonzero"], 2)
        self.assertEqual(r.n, 2)

    def test_all_zero_differences_is_p_one_not_a_crash(self):
        r = wilcoxon_signed_rank([0.3] * 5, [0.3] * 5)
        self.assertEqual(r.pvalue, 1.0)
        self.assertEqual(r.n, 0)
        self.assertTrue(math.isnan(r.statistic))

    def test_consistent_shift_is_significant(self):
        a = [0.1 * i for i in range(1, 21)]
        b = [v - 0.05 for v in a]
        r = wilcoxon_signed_rank(a, b)
        self.assertLess(r.pvalue, 0.001)
        self.assertEqual(r.detail["n_positive"], 20)
        self.assertEqual(r.detail["n_negative"], 0)

    def test_symmetric_noise_is_not_significant(self):
        a = [0.5, 0.4, 0.6, 0.45, 0.55, 0.42, 0.58, 0.48]
        b = [0.4, 0.5, 0.45, 0.6, 0.42, 0.55, 0.48, 0.58]
        self.assertGreater(wilcoxon_signed_rank(a, b).pvalue, 0.05)

    def test_ties_use_normal_approximation_branch(self):
        a = [0.2, 0.2, 0.2, -0.2, 0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        b = [0.0] * len(a)
        r = wilcoxon_signed_rank(a, b)
        self.assertIn("normal approximation", r.detail["method"])
        self.assertTrue(0.0 <= r.pvalue <= 1.0)

    def test_large_n_uses_approximation_and_stays_in_range(self):
        a = [math.sin(i) for i in range(120)]
        b = [0.0] * 120
        r = wilcoxon_signed_rank(a, b)
        self.assertIn("normal approximation", r.detail["method"])
        self.assertTrue(0.0 <= r.pvalue <= 1.0)

    def test_nan_pairs_dropped(self):
        a = [0.1, float("nan"), 0.3, 0.4]
        b = [0.0, 0.0, 0.0, float("nan")]
        r = wilcoxon_signed_rank(a, b)
        self.assertEqual(r.detail["n_pairs"], 2)


class TestHolm(unittest.TestCase):
    def test_known_worked_example(self):
        """p = [0.01, 0.04, 0.03] -> adjusted [0.03, 0.06, 0.06]."""
        out = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(out["a"]["p_adjusted"], 0.03, places=12)
        self.assertAlmostEqual(out["b"]["p_adjusted"], 0.06, places=12)
        self.assertAlmostEqual(out["c"]["p_adjusted"], 0.06, places=12)
        self.assertTrue(out["a"]["reject"])
        self.assertFalse(out["b"]["reject"])
        self.assertFalse(out["c"]["reject"])

    def test_monotonicity_enforced(self):
        out = holm_bonferroni({"a": 0.02, "b": 0.021, "c": 0.9, "d": 0.03, "e": 0.04})
        adjusted = sorted(
            (v["p_raw"], v["p_adjusted"]) for v in out.values()
        )
        values = [adj for _, adj in adjusted]
        self.assertEqual(values, sorted(values))

    def test_capped_at_one(self):
        out = holm_bonferroni({"a": 0.5, "b": 0.6, "c": 0.7, "d": 0.8, "e": 0.9})
        for entry in out.values():
            self.assertLessEqual(entry["p_adjusted"], 1.0)

    def test_never_less_than_raw_p(self):
        out = holm_bonferroni({"a": 0.001, "b": 0.01, "c": 0.2, "d": 0.4, "e": 0.05})
        for entry in out.values():
            self.assertGreaterEqual(entry["p_adjusted"] + 1e-15, entry["p_raw"])

    def test_single_test_is_unchanged(self):
        out = holm_bonferroni({"only": 0.042})
        self.assertAlmostEqual(out["only"]["p_adjusted"], 0.042, places=12)

    def test_list_input_returns_list_in_order(self):
        out = holm_bonferroni([0.01, 0.04, 0.03])
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0], 0.03, places=12)
        self.assertAlmostEqual(out[1], 0.06, places=12)
        self.assertAlmostEqual(out[2], 0.06, places=12)

    def test_empty_and_nan_inputs(self):
        self.assertEqual(holm_bonferroni({}), {})
        out = holm_bonferroni({"a": float("nan"), "b": 0.01})
        self.assertNotIn("a", out)
        self.assertAlmostEqual(out["b"]["p_adjusted"], 0.01, places=12)


class TestCompareArms(unittest.TestCase):
    def _arms(self):
        agentic = {
            "exact_match": [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            "similarity": [0.5, 0.4, 0.6, 0.55, 0.48, 0.62, 0.51, 0.44, 0.59, 0.47],
            "adjacent_pair_acc": [0.4, 0.35, 0.5, 0.45, 0.38, 0.52, 0.41, 0.34, 0.49, 0.37],
            "longest_correct_run": [0.3, 0.25, 0.4, 0.35, 0.28, 0.42, 0.31, 0.24, 0.39, 0.27],
            "kendall_tau": [0.3, 0.2, 0.4, 0.35, 0.25, 0.45, 0.32, 0.22, 0.38, 0.28],
        }
        control = {
            k: ([0, 0, 0, 1, 0, 0, 0, 0, 1, 0] if k == "exact_match" else [v - 0.02 for v in vals])
            for k, vals in agentic.items()
        }
        return agentic, control

    def test_returns_all_five_metrics_with_tests_and_cis(self):
        agentic, control = self._arms()
        out = compare_arms(agentic, control)
        self.assertEqual(set(out["metrics"]), set(agentic))
        self.assertEqual(out["family_size"], 5)
        for metric, entry in out["metrics"].items():
            self.assertIn("test", entry)
            self.assertIn("delta_ci", entry)
            self.assertIn("holm", entry)
            self.assertTrue(0.0 <= entry["test"]["pvalue"] <= 1.0)

    def test_exact_match_routed_to_mcnemar_others_to_wilcoxon(self):
        agentic, control = self._arms()
        out = compare_arms(agentic, control)
        self.assertIn("mcnemar", out["metrics"]["exact_match"]["test"]["name"])
        self.assertIn("discordant", out["metrics"]["exact_match"]["test"]["detail"])
        for metric in ("similarity", "adjacent_pair_acc", "longest_correct_run", "kendall_tau"):
            self.assertIn("wilcoxon", out["metrics"][metric]["test"]["name"])
            self.assertIn("n_nonzero", out["metrics"][metric]["test"]["detail"])

    def test_deterministic_across_calls(self):
        agentic, control = self._arms()
        self.assertEqual(compare_arms(agentic, control), compare_arms(agentic, control))

    def test_consistent_advantage_survives_holm(self):
        agentic, control = self._arms()
        out = compare_arms(agentic, control)
        # every continuous metric is +0.02 on every sample -> should reject
        for metric in ("similarity", "adjacent_pair_acc", "longest_correct_run", "kendall_tau"):
            self.assertTrue(out["metrics"][metric]["holm"]["reject"], metric)


if __name__ == "__main__":
    unittest.main(verbosity=2)
