"""Unit and regression tests for evaluation/analysis.py.

    python tests/test_analysis.py

The regression tests parse the benchmark table out of each existing report.md
and require the values recomputed from samples.jsonl to match exactly, so the
report generator cannot drift from the reports already written."""

from __future__ import annotations

import math
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.analysis import (  # noqa: E402
    ARMS,
    METRIC_KEYS,
    TAXONOMY_THRESHOLDS,
    bin_labels,
    breakpoints,
    breakpoint_stats,
    classify_error,
    cost_summary,
    discover_runs,
    fragment_bin,
    fragment_count,
    junction_ranking_summary,
    lift_over_baseline,
    load_run,
    n_terminal_correct,
    nterm_analysis,
    oracle_gap,
    sample_rows,
    stratify_by_bin,
    trypsin_recall_summary,
)
from evaluation.metrics import METRIC_NAMES, nanmean  # noqa: E402

RUNS = discover_runs()

# Column order of the benchmark table emitted by evaluation/reporting.py.
REPORT_COLUMNS = ["shuffled", "deterministic", "control", "agentic", "oracle"]


def _make_sample(**overrides):
    """A minimal sample record with all five arms present."""
    base = {
        "index": 1,
        "target": "M" * 100,
        "order": list(range(10)),
        "baseline_order": list(range(10))[::-1],
    }
    for arm, key in ARMS.items():
        base[key] = {
            "exact_match": 0.0,
            "similarity": 0.3,
            "adjacent_pair_acc": 0.4,
            "longest_correct_run": 0.3,
            "kendall_tau": 0.35,
            "true_order_recovered": True,
        }
    base.update(overrides)
    return base


class TestFragmentBinning(unittest.TestCase):
    def test_bin_boundaries_are_inclusive_and_contiguous(self):
        self.assertEqual(fragment_bin(2), "2-4")
        self.assertEqual(fragment_bin(4), "2-4")
        self.assertEqual(fragment_bin(5), "5-9")
        self.assertEqual(fragment_bin(9), "5-9")
        self.assertEqual(fragment_bin(10), "10-19")
        self.assertEqual(fragment_bin(19), "10-19")
        self.assertEqual(fragment_bin(20), "20-49")
        self.assertEqual(fragment_bin(49), "20-49")
        self.assertEqual(fragment_bin(50), "50+")
        self.assertEqual(fragment_bin(5000), "50+")

    def test_bin_labels_match_binning(self):
        self.assertEqual(bin_labels(), ["2-4", "5-9", "10-19", "20-49", "50+"])

    def test_fragment_count_from_order(self):
        self.assertEqual(fragment_count({"order": [0, 1, 2, 3]}), 4)

    def test_fragment_count_falls_back_to_total_junctions(self):
        # total = n(n-1); 1056 -> 33
        self.assertEqual(fragment_count({"total_junctions": 1056}), 33)

    def test_stratify_retains_empty_bins(self):
        rows = [{"fragment_bin": "5-9", "agentic_similarity": 0.5}]
        out = stratify_by_bin(rows, "agentic", "similarity")
        self.assertEqual(set(out), set(bin_labels()))
        self.assertEqual(out["2-4"]["n"], 0)
        self.assertEqual(out["5-9"]["values"], [0.5])


class TestDerivedQuantities(unittest.TestCase):
    def test_n_terminal_correct_uses_identity_true_order(self):
        self.assertTrue(n_terminal_correct([0, 5, 3]))
        self.assertFalse(n_terminal_correct([2, 0, 1]))
        self.assertIsNone(n_terminal_correct([]))

    def test_breakpoints_is_n_minus_1_times_one_minus_apa(self):
        sample = _make_sample(order=list(range(11)))  # n = 11 -> 10 joins
        sample["recon_metrics"]["adjacent_pair_acc"] = 0.8
        self.assertAlmostEqual(breakpoints(sample), 2.0, places=12)

    def test_breakpoints_integral_for_realistic_apa(self):
        """APA is correct_adjacencies/(n-1), so (n-1)*(1-APA) must come back to a
        whole number of wrong joins."""
        for n in (5, 12, 33, 150):
            for correct in (0, 1, n // 2, n - 1):
                sample = _make_sample(order=list(range(n)))
                sample["recon_metrics"]["adjacent_pair_acc"] = correct / (n - 1)
                value = breakpoints(sample)
                self.assertAlmostEqual(value, (n - 1) - correct, places=9)

    def test_breakpoints_none_when_metrics_are_nan(self):
        sample = _make_sample()
        sample["recon_metrics"]["adjacent_pair_acc"] = float("nan")
        self.assertIsNone(breakpoints(sample))

    def test_perfect_ordering_has_zero_breakpoints(self):
        sample = _make_sample()
        sample["recon_metrics"]["adjacent_pair_acc"] = 1.0
        self.assertAlmostEqual(breakpoints(sample), 0.0, places=12)


class TestTaxonomy(unittest.TestCase):
    def _classify(self, **metrics):
        sample = _make_sample(order=metrics.pop("order", list(range(11))))
        sample["recon_metrics"].update(metrics)
        return classify_error(sample)

    def test_exact_match_wins_over_everything(self):
        self.assertEqual(
            self._classify(exact_match=1.0, adjacent_pair_acc=1.0, kendall_tau=1.0), "exact"
        )

    def test_block_reversal(self):
        self.assertEqual(
            self._classify(kendall_tau=-0.9, adjacent_pair_acc=0.0, longest_correct_run=0.1),
            "block_reversal",
        )

    def test_full_scramble(self):
        self.assertEqual(
            self._classify(kendall_tau=0.02, adjacent_pair_acc=0.0, longest_correct_run=0.1),
            "full_scramble",
        )

    def test_local_transposition_needs_few_breakpoints_and_a_long_run(self):
        # n=11 -> 10 joins; APA 0.9 -> 1 breakpoint, LCR 0.6
        self.assertEqual(
            self._classify(adjacent_pair_acc=0.9, longest_correct_run=0.6, kendall_tau=0.8),
            "local_transposition",
        )

    def test_wrong_start_when_structured_but_misanchored(self):
        self.assertEqual(
            self._classify(
                order=[3, 0, 1, 2] + list(range(4, 11)),
                adjacent_pair_acc=0.7,
                longest_correct_run=0.4,
                kendall_tau=0.6,
            ),
            "wrong_start",
        )

    def test_reversal_and_scramble_outrank_a_wrong_start(self):
        """A reversal or scramble is a failure SHAPE that subsumes a wrong
        start, so those must win even when order[0] != 0."""
        misanchored = [3, 0, 1, 2] + list(range(4, 11))
        self.assertEqual(
            self._classify(
                order=misanchored, kendall_tau=-0.9, adjacent_pair_acc=0.0,
                longest_correct_run=0.1,
            ),
            "block_reversal",
        )
        self.assertEqual(
            self._classify(
                order=misanchored, kendall_tau=0.02, adjacent_pair_acc=0.0,
                longest_correct_run=0.1,
            ),
            "full_scramble",
        )

    def test_partial_assembly_is_the_correct_start_fallback(self):
        self.assertEqual(
            self._classify(adjacent_pair_acc=0.3, longest_correct_run=0.2, kendall_tau=0.4),
            "partial_assembly",
        )

    def test_nan_metrics_are_unknown_not_silently_binned(self):
        self.assertEqual(self._classify(kendall_tau=float("nan")), "unknown")

    def test_every_sample_gets_exactly_one_class(self):
        """No real sample may fall through the chain unlabelled."""
        for run_dir in RUNS:
            rows = sample_rows(load_run(run_dir))
            for row in rows:
                self.assertIn(
                    row["error_class"],
                    set(TAXONOMY_THRESHOLDS) | {
                        "exact", "block_reversal", "full_scramble",
                        "local_transposition", "wrong_start", "partial_assembly", "unknown",
                    },
                )

    def test_thresholds_are_documented_constants(self):
        for key in (
            "reversal_tau_max",
            "scramble_tau_abs_max",
            "scramble_apa_max",
            "local_max_breakpoints",
            "local_min_lcr",
        ):
            self.assertIn(key, TAXONOMY_THRESHOLDS)


class TestLift(unittest.TestCase):
    def test_lift_is_paired_not_a_difference_of_means(self):
        rows = [
            {"fragment_bin": "5-9", "agentic_similarity": 0.5, "shuffled_similarity": 0.2},
            {"fragment_bin": "5-9", "agentic_similarity": 0.9, "shuffled_similarity": 0.8},
        ]
        out = lift_over_baseline(rows, "similarity")["5-9"]
        self.assertAlmostEqual(out["lift"], (0.3 + 0.1) / 2, places=12)
        for got, want in zip(out["deltas"], [0.3, 0.1]):
            self.assertAlmostEqual(got, want, places=12)

    def test_lift_skips_unusable_pairs_but_counts_them(self):
        rows = [
            {"fragment_bin": "5-9", "agentic_similarity": 0.5, "shuffled_similarity": 0.2},
            {"fragment_bin": "5-9", "agentic_similarity": float("nan"), "shuffled_similarity": 0.1},
        ]
        out = lift_over_baseline(rows, "similarity")["5-9"]
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["n_usable"], 1)


class TestOptionalFieldsDegradeGracefully(unittest.TestCase):
    def test_missing_junction_ranking_returns_none_not_zero(self):
        rows = sample_rows(load_run(RUNS[0])) if RUNS else []
        self.assertIsNone(junction_ranking_summary(rows))

    def test_missing_trypsin_recall_returns_none_not_zero(self):
        rows = sample_rows(load_run(RUNS[0])) if RUNS else []
        self.assertIsNone(trypsin_recall_summary(rows))

    def test_present_junction_ranking_is_aggregated(self):
        rows = [
            {"junction_top1_acc": 0.4, "junction_top3_acc": 0.7, "junction_mrr": 0.55,
             "junction_num_junctions": 30},
            {"junction_top1_acc": 0.6, "junction_top3_acc": 0.9, "junction_mrr": 0.75,
             "junction_num_junctions": 20},
        ]
        out = junction_ranking_summary(rows)
        self.assertAlmostEqual(out["top1_acc"], 0.5, places=12)
        self.assertEqual(out["total_junctions"], 50)


@unittest.skipIf(not RUNS, "no runs in results/")
class TestRegressionAgainstShippedReports(unittest.TestCase):
    """The contract: numbers recomputed from samples.jsonl must equal the
    numbers already printed in each run's report.md, to the printed precision."""

    @staticmethod
    def _parse_benchmark_table(report_path: Path) -> dict:
        """Pull {metric_label: [col values]} out of the benchmark table."""
        text = report_path.read_text(encoding="utf-8")
        labels = set(METRIC_NAMES.values())
        found = {}
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] in labels and cells[0] not in found:
                numeric = []
                for cell in cells[1:]:
                    try:
                        numeric.append(float(cell.replace("+", "")))
                    except ValueError:
                        break
                found[cells[0]] = numeric
        return found

    def test_all_arms_reproduce_report_tables_exactly(self):
        label_to_key = {v: k for k, v in METRIC_NAMES.items()}
        checked = 0
        for run_dir in RUNS:
            report = run_dir / "report.md"
            if not report.exists():
                continue
            run = load_run(run_dir)
            rows = sample_rows(run)
            table = self._parse_benchmark_table(report)
            self.assertTrue(table, f"no benchmark table parsed from {report}")

            for label, printed in table.items():
                metric = label_to_key[label]
                for column, arm in enumerate(REPORT_COLUMNS):
                    if arm not in run.arms or column >= len(printed):
                        continue
                    recomputed = nanmean([r.get(f"{arm}_{metric}") for r in rows])
                    self.assertEqual(
                        f"{recomputed:.4f}",
                        f"{printed[column]:.4f}",
                        msg=f"{run_dir.name} {label} [{arm}]",
                    )
                    checked += 1
        self.assertGreater(checked, 100, "regression test did not actually check much")

    def test_sample_counts_match_summary(self):
        for run_dir in RUNS:
            run = load_run(run_dir)
            if run.summary.get("sample_count") is not None:
                self.assertEqual(run.n, run.summary["sample_count"], run_dir.name)

    def test_every_row_has_all_metrics_for_every_arm(self):
        for run_dir in RUNS:
            run = load_run(run_dir)
            rows = sample_rows(run)
            for row in rows:
                for arm in run.arms:
                    for metric in METRIC_KEYS:
                        self.assertIn(f"{arm}_{metric}", row)

    def test_derived_quantities_are_populated_on_real_data(self):
        for run_dir in RUNS:
            rows = sample_rows(load_run(run_dir))
            nterm = nterm_analysis(rows)
            self.assertEqual(nterm["n"], len(rows))
            self.assertIsNotNone(nterm["p_correct_start"])
            bp = breakpoint_stats(rows)
            self.assertGreater(bp["n"], 0)
            self.assertGreaterEqual(bp["min"], 0.0)
            cost = cost_summary(rows)
            self.assertEqual(cost["n_samples"], len(rows))

    def test_oracle_never_below_agentic(self):
        """The oracle selects among generated candidates, so by construction it
        cannot be worse than the arm that also selected from that set."""
        for run_dir in RUNS:
            run = load_run(run_dir)
            if "oracle" not in run.arms:
                continue
            for metric, stats in oracle_gap(sample_rows(run)).items():
                self.assertGreaterEqual(
                    stats["mean_gap"], -1e-12, f"{run_dir.name} {metric}"
                )

    def test_breakpoints_are_whole_numbers_on_real_data(self):
        for run_dir in RUNS:
            for row in sample_rows(load_run(run_dir)):
                value = row.get("breakpoints")
                if isinstance(value, (int, float)) and not math.isnan(value):
                    self.assertAlmostEqual(value, round(value), places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
