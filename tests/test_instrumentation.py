"""Tests for evaluation/instrumentation.py and the end-of-run reporting hook.

    python tests/test_instrumentation.py

The contract: the diagnostics are additive and never change an experiment
result, they close the junction-ranking and trypsin-recall gaps, they can never
raise into the run loop, and reports built from older runs that lack the fields
still work."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.analysis import (  # noqa: E402
    junction_ranking_summary,
    sample_rows,
    trypsin_recall_summary,
)
from evaluation.instrumentation import safe_sample_diagnostics, sample_diagnostics  # noqa: E402
from evaluation.metrics import junction_ranking_stats  # noqa: E402


class TestDiagnostics(unittest.TestCase):
    """A 4-fragment protein whose true order is the identity permutation."""

    def setUp(self):
        self.fragments = ["MAK", "LVR", "PPK", "GG"]
        self.target = "".join(self.fragments)
        # A score matrix that ranks the true successor top for every junction.
        n = len(self.fragments)
        self.scores = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                self.scores[i][j] = 1.0 if j == i + 1 else 0.1

    def test_records_fragments_and_true_order(self):
        out = sample_diagnostics(self.target, self.fragments, {})
        self.assertEqual(out["fragments"], self.fragments)
        self.assertEqual(out["true_order"], [0, 1, 2, 3])

    def test_junction_ranking_is_recorded(self):
        out = sample_diagnostics(
            self.target, self.fragments, {"scores": self.scores}
        )
        ranking = out["junction_ranking"]
        self.assertAlmostEqual(ranking["top1_acc"], 1.0, places=12)
        self.assertAlmostEqual(ranking["mrr"], 1.0, places=12)
        self.assertEqual(ranking["num_junctions"], 3)

    def test_confirmed_junctions_are_excluded_from_ranking(self):
        """Confirmed adjacencies are handed to the search as known, so counting
        them would inflate the pLM's apparent accuracy."""
        out = sample_diagnostics(
            self.target,
            self.fragments,
            {"scores": self.scores, "confirmed_junctions": {(0, 1)}},
        )
        ranking = out["junction_ranking"]
        self.assertEqual(ranking["num_junctions"], 2)
        self.assertEqual(ranking["excluded_confirmed"], 1)

    def test_trypsin_recall_detects_a_wrongly_pruned_true_junction(self):
        """The measurement the pruned COUNT alone could never provide."""
        out = sample_diagnostics(
            self.target, self.fragments, {"impossible_junctions": {(1, 2), (3, 0)}}
        )
        recall = out["trypsin_recall"]
        self.assertEqual(recall["true_junctions"], 3)
        self.assertEqual(recall["true_junctions_pruned"], 1)  # (1,2) is a true join
        self.assertAlmostEqual(recall["recall"], 2 / 3, places=12)

    def test_perfect_filter_has_recall_one(self):
        out = sample_diagnostics(
            self.target, self.fragments, {"impossible_junctions": {(3, 0), (2, 0)}}
        )
        self.assertAlmostEqual(out["trypsin_recall"]["recall"], 1.0, places=12)

    def test_untileable_fragments_degrade_without_raising(self):
        out = sample_diagnostics("MAKLVR", ["QQQ", "ZZZ"], {"scores": [[0, 1], [1, 0]]})
        self.assertIsNone(out["true_order"])
        self.assertNotIn("junction_ranking", out)

    def test_empty_fragments(self):
        self.assertEqual(sample_diagnostics("MAK", [], {}), {})

    def test_safe_wrapper_never_raises(self):
        out = safe_sample_diagnostics(self.target, self.fragments, {"scores": "not a matrix"})
        self.assertIn("diagnostics_error", out)

    def test_safe_wrapper_handles_none_state(self):
        out = safe_sample_diagnostics(self.target, self.fragments, None)
        self.assertEqual(out["true_order"], [0, 1, 2, 3])


class TestSkipPairsIsBackwardsCompatible(unittest.TestCase):
    def test_default_behaviour_unchanged(self):
        """junction_ranking_stats gained an optional parameter; the standalone
        dense diagnostic must be unaffected."""
        scores = [[0.0, 1.0, 0.1], [0.1, 0.0, 1.0], [0.1, 0.1, 0.0]]
        without = junction_ranking_stats(scores, [0, 1, 2], 3)
        explicit_none = junction_ranking_stats(scores, [0, 1, 2], 3, skip_pairs=None)
        self.assertEqual(without, explicit_none)
        self.assertEqual(without["num_junctions"], 2)

    def test_empty_skip_set_is_a_no_op(self):
        scores = [[0.0, 1.0, 0.1], [0.1, 0.0, 1.0], [0.1, 0.1, 0.0]]
        self.assertEqual(
            junction_ranking_stats(scores, [0, 1, 2], 3),
            junction_ranking_stats(scores, [0, 1, 2], 3, skip_pairs=set()),
        )


class TestReportingConsumesOptionalFields(unittest.TestCase):
    """A synthetic run folder proves the new fields flow all the way through to
    the report aggregations, without needing to execute a real run."""

    def _write_run(self, directory: Path, with_diagnostics: bool):
        fragments = ["MAK", "LVR", "PPK", "GG"]
        target = "".join(fragments)
        sample = {
            "index": 1,
            "target": target,
            "reconstruction": target,
            "order": [0, 1, 2, 3],
            "baseline_order": [3, 2, 1, 0],
            "baseline_metrics": {
                "exact_match": 0.0, "similarity": 0.1, "adjacent_pair_acc": 0.0,
                "longest_correct_run": 0.25, "kendall_tau": -1.0,
                "true_order_recovered": True,
            },
            "recon_metrics": {
                "exact_match": 1.0, "similarity": 1.0, "adjacent_pair_acc": 1.0,
                "longest_correct_run": 1.0, "kendall_tau": 1.0,
                "true_order_recovered": True,
            },
            "completed": True,
        }
        if with_diagnostics:
            sample.update(
                sample_diagnostics(
                    target,
                    fragments,
                    {
                        "scores": [[1.0 if j == i + 1 else 0.1 for j in range(4)] for i in range(4)],
                        "impossible_junctions": {(3, 0)},
                    },
                )
            )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "samples.jsonl").write_text(
            json.dumps(sample) + "\n", encoding="utf-8"
        )
        (directory / "summary.json").write_text(
            json.dumps({"run_name": "synthetic", "config": {}, "sample_count": 1}),
            encoding="utf-8",
        )

    def test_new_fields_reach_the_summaries(self):
        from evaluation.analysis import load_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_new"
            self._write_run(run_dir, with_diagnostics=True)
            rows = sample_rows(load_run(run_dir))
            ranking = junction_ranking_summary(rows)
            recall = trypsin_recall_summary(rows)
            self.assertIsNotNone(ranking)
            self.assertAlmostEqual(ranking["top1_acc"], 1.0, places=12)
            self.assertIsNotNone(recall)
            self.assertAlmostEqual(recall["recall"], 1.0, places=12)

    def test_old_runs_without_the_fields_still_load(self):
        from evaluation.analysis import load_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_old"
            self._write_run(run_dir, with_diagnostics=False)
            rows = sample_rows(load_run(run_dir))
            self.assertEqual(len(rows), 1)
            self.assertIsNone(junction_ranking_summary(rows))
            self.assertIsNone(trypsin_recall_summary(rows))

    def test_full_report_generates_for_both(self):
        """End to end: the generator must produce a report for a run with the
        new fields and for one without."""
        from evaluation.rebuild import generate_run_artifacts

        for with_diagnostics in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run"
                self._write_run(run_dir, with_diagnostics=with_diagnostics)
                result = generate_run_artifacts(run_dir, resamples=50, quiet=True)
                text = result["report"].read_text(encoding="utf-8")
                self.assertIn("## A. Overall Performance", text)
                self.assertIn("## G. Cost", text)
                self.assertTrue(result["results_csv"].exists())
                self.assertTrue(result["summary_csv"].exists())
                if with_diagnostics:
                    self.assertIn("Top-1 successor accuracy", text)
                else:
                    self.assertIn("n/a - requires field", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
