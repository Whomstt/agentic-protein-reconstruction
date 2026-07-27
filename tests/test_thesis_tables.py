"""Independent verification of the report's Results tables.

Every check here recomputes a value **from scratch** out of the run's
``samples.jsonl`` and asserts that the exact formatted string appears in the
emitted ``.tex``. The point is independence, so this file deliberately imports
nothing from ``evaluation/`` — no shared aggregation, no shared formatter, no
shared statistics. Only ``json``, ``math``, ``statistics`` and the stdlib
``unittest``. If a helper in ``evaluation/`` were wrong, importing it here would
hide the error rather than catch it.

Three or more values are verified per table, at the precision the table prints.

    python tests/test_thesis_tables.py

Regenerate the tables first if the run data changed:

    python -m evaluation.thesis_tables --run 130726_224804_agentic
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = "130726_224804_agentic"
SAMPLES = ROOT / "results" / RUN / "samples.jsonl"
TABLES = ROOT / "report" / "tables"

ARM_KEY = {
    "shuffled": "baseline_metrics",
    "deterministic": "first_pass_metrics",
    "control": "control_metrics",
    "agentic": "recon_metrics",
    "oracle": "oracle_metrics",
}


def load():
    with SAMPLES.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tex(name: str) -> str:
    return (TABLES / f"{name}.tex").read_text(encoding="utf-8")


def parse_table(name: str):
    """Parse an emitted booktabs table back into (headers, rows-by-first-cell).

    Verifying a bare substring would let a value match some *other* cell that
    happens to print the same digits, so the grid checks below compare the exact
    cell at a named (row, column) instead.
    """
    lines = tex(name).splitlines()
    start = lines.index(r"    \toprule")
    mid = lines.index(r"    \midrule")
    end = lines.index(r"    \bottomrule")

    def split(line):
        return [c.strip() for c in line.strip().removesuffix(r"\\").strip().split("&")]

    headers = split(lines[start + 1])
    rows = {}
    for line in lines[mid + 1 : end]:
        cells = split(line)
        rows[cells[0]] = dict(zip(headers, cells))
    return headers, rows


def values(samples, arm, metric):
    """Usable per-sample values (None/NaN dropped), independent of analysis.py."""
    out = []
    for s in samples:
        v = (s.get(ARM_KEY[arm]) or {}).get(metric)
        if isinstance(v, (int, float)) and not math.isnan(v):
            out.append(v)
    return out


def mean(xs):
    return sum(xs) / len(xs)


def paired(samples, arm_a, arm_b, metric):
    out = []
    for s in samples:
        a = (s.get(ARM_KEY[arm_a]) or {}).get(metric)
        b = (s.get(ARM_KEY[arm_b]) or {}).get(metric)
        if (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and not math.isnan(a)
            and not math.isnan(b)
        ):
            out.append((a, b))
    return out


def f3(x):
    return f"{x:.3f}"


def f3s(x):
    return f"{x:+.3f}"


def nfrag(sample):
    return len(sample.get("order") or [])


def frag_bin(n):
    if 2 <= n <= 4:
        return "2-4"
    if 5 <= n <= 9:
        return "5-9"
    if 10 <= n <= 19:
        return "10-19"
    if 20 <= n <= 49:
        return "20-49"
    return "50+" if n >= 50 else "<2"


class ThesisTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SAMPLES.exists():
            raise unittest.SkipTest(f"{SAMPLES} not found")
        cls.samples = load()

    def assertInTex(self, table, needle, what):
        self.assertIn(
            needle,
            tex(table),
            f"{what}: independently computed {needle!r} is not present in {table}.tex",
        )

    def assertCell(self, table, row_label, column, expected, what):
        """Exact equality against the cell at (row, column) in the emitted .tex."""
        _, rows = parse_table(table)
        self.assertIn(row_label, rows, f"{what}: no row {row_label!r} in {table}.tex")
        actual = rows[row_label][column]
        self.assertEqual(
            actual,
            expected,
            f"{what}: {table}.tex row {row_label!r} column {column!r} prints "
            f"{actual!r}, independently computed {expected!r}",
        )

    def assertCellStartsWith(self, table, row_label, column, expected, what):
        """For a cell of the form 'point [low, high]' whose interval is a BCa
        bootstrap: the point estimate is recomputed here, the interval bounds are
        not (reimplementing BCa would not be an independent check of it)."""
        _, rows = parse_table(table)
        actual = rows[row_label][column]
        self.assertTrue(
            actual.startswith(expected + " ["),
            f"{what}: {table}.tex row {row_label!r} column {column!r} prints "
            f"{actual!r}, independently computed point estimate {expected!r}",
        )

    # --- I. main results --------------------------------------------------
    def test_main_results(self):
        s = self.samples
        self.assertCellStartsWith(
            "thesis_main_results",
            "Adjacent Pair Accuracy",
            "Agentic",
            f3(mean(values(s, "agentic", "adjacent_pair_acc"))),
            "Agentic Adjacent Pair Accuracy",
        )
        self.assertCellStartsWith(
            "thesis_main_results",
            "Sequence Similarity",
            "Deterministic",
            f3(mean(values(s, "deterministic", "similarity"))),
            "Deterministic Sequence Similarity",
        )
        self.assertCellStartsWith(
            "thesis_main_results",
            "Kendall Tau",
            "Shuffled Baseline",
            f3(mean(values(s, "shuffled", "kendall_tau"))),
            "Shuffled Kendall Tau",
        )
        # The Wilson interval for Exact Match, reimplemented from its closed form.
        em = values(s, "agentic", "exact_match")
        successes, n = sum(1 for v in em if v >= 1.0), len(em)
        z = 1.959963984540054  # two-sided 95%
        p = successes / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        self.assertCell(
            "thesis_main_results",
            "Exact Match",
            "Agentic",
            f"{f3(p)} [{f3(center - margin)}, {f3(center + margin)}]",
            "Agentic Exact Match, point and full Wilson interval",
        )

    # --- II. paired tests -------------------------------------------------
    def test_paired_tests(self):
        s = self.samples
        pairs = paired(s, "agentic", "control", "adjacent_pair_acc")
        self.assertInTex(
            "thesis_paired_tests",
            f3s(mean([a - b for a, b in pairs])),
            "Mean delta, Agentic - Control, Adjacent Pair Accuracy",
        )

        # McNemar discordant counts on Exact Match, Agentic vs Control.
        em = paired(s, "agentic", "control", "exact_match")
        only_a = sum(1 for a, b in em if a >= 1.0 > b)
        only_b = sum(1 for a, b in em if b >= 1.0 > a)
        self.assertInTex(
            "thesis_paired_tests",
            f"{only_a + only_b} ({only_a}/{only_b})",
            "McNemar discordant pairs, Exact Match, Agentic - Control",
        )

        # ...and its exact two-sided binomial p-value.
        k, n = min(only_a, only_b), only_a + only_b
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
        self.assertInTex(
            "thesis_paired_tests",
            f"{min(1.0, 2 * tail):.3f}",
            "McNemar exact p, Exact Match, Agentic - Control",
        )

        # Wilcoxon non-zero pair counts, Kendall Tau, Agentic vs Deterministic.
        tau = paired(s, "agentic", "deterministic", "kendall_tau")
        deltas = [a - b for a, b in tau if a != b]
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        self.assertInTex(
            "thesis_paired_tests",
            f"{len(deltas)} ({pos}/{neg})",
            "Wilcoxon non-zero pairs, Kendall Tau, Agentic - Deterministic",
        )

    # --- III. stratification ----------------------------------------------
    def test_stratification(self):
        s = self.samples
        by_bin = {}
        for sample in s:
            by_bin.setdefault(frag_bin(nfrag(sample)), []).append(sample)

        self.assertCell(
            "thesis_stratification", "10-19", "n", str(len(by_bin["10-19"])),
            "Proteins in the 10-19 fragment bin",
        )

        def arm_mean(bin_label, arm):
            vals = [
                (x.get(ARM_KEY[arm]) or {}).get("adjacent_pair_acc")
                for x in by_bin[bin_label]
            ]
            return mean([v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)])

        self.assertCell(
            "thesis_stratification",
            "20-49",
            "Agentic",
            f3(arm_mean("20-49", "agentic")),
            "Agentic APA in the 20-49 bin",
        )

        lifts = []
        for sample in by_bin["5-9"]:
            a = (sample.get("recon_metrics") or {}).get("adjacent_pair_acc")
            b = (sample.get("baseline_metrics") or {}).get("adjacent_pair_acc")
            if all(isinstance(v, (int, float)) and not math.isnan(v) for v in (a, b)):
                lifts.append(a - b)
        self.assertCell(
            "thesis_stratification", "5-9", "Lift", f3s(mean(lifts)),
            "Paired lift over the shuffled floor in the 5-9 bin",
        )

    # --- IV. selection ceiling --------------------------------------------
    def test_selection_ceiling(self):
        s = self.samples
        self.assertCell(
            "thesis_selection_ceiling",
            "Sequence Similarity",
            "Oracle",
            f3(mean(values(s, "oracle", "similarity"))),
            "Oracle Sequence Similarity",
        )
        tau = paired(s, "oracle", "agentic", "kendall_tau")
        self.assertCell(
            "thesis_selection_ceiling",
            "Kendall Tau",
            "Gap",
            f3s(mean([o - a for o, a in tau])),
            "Mean oracle gap, Kendall Tau",
        )
        apa = paired(s, "oracle", "agentic", "adjacent_pair_acc")
        with_gap = sum(1 for o, a in apa if o - a > 1e-12)
        self.assertCell(
            "thesis_selection_ceiling",
            "Adjacent Pair Accuracy",
            "Samples with a gap",
            f"{with_gap}/{len(apa)}",
            "Samples with an Adjacent Pair Accuracy gap",
        )

    # --- V. validity concordance ------------------------------------------
    def test_validity_concordance(self):
        """Concordance recomputed from the definition: over candidate pairs
        within a sample, how often does the lower validity score go with the
        higher true quality? Ties on either axis are not comparable."""
        per_sample = []
        total_pairs = 0
        for sample in self.samples:
            points = []
            for record in sample.get("iteration_history") or []:
                v = record.get("validity_score")
                q = (record.get("metrics") or {}).get("adjacent_pair_acc")
                if (
                    isinstance(v, (int, float))
                    and isinstance(q, (int, float))
                    and not math.isnan(v)
                    and not math.isnan(q)
                    and not math.isinf(v)
                ):
                    points.append((v, q))
            good = comparable = 0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    (v1, q1), (v2, q2) = points[i], points[j]
                    if v1 == v2 or q1 == q2:
                        continue
                    comparable += 1
                    if (v1 < v2 and q1 > q2) or (v2 < v1 and q2 > q1):
                        good += 1
            total_pairs += comparable
            if comparable:
                per_sample.append(good / comparable)

        self.assertCell(
            "thesis_validity_concordance",
            "Samples with comparable candidate pairs",
            "Value",
            str(len(per_sample)),
            "Number of samples with comparable candidates",
        )
        self.assertCell(
            "thesis_validity_concordance",
            "Comparable candidate pairs",
            "Value",
            str(total_pairs),
            "Total comparable candidate pairs",
        )
        self.assertCell(
            "thesis_validity_concordance",
            "Mean within-sample concordance",
            "Value",
            f3(mean(per_sample)),
            "Mean within-sample concordance",
        )
        above = sum(1 for c in per_sample if c > 0.5)
        self.assertCell(
            "thesis_validity_concordance",
            "Samples above chance (> 0.50)",
            "Value",
            f"{above}/{len(per_sample)} ({100.0 * above / len(per_sample):.1f}\\%)",
            "Samples above chance",
        )

    # --- VI. agent behaviour ----------------------------------------------
    def test_agent_behaviour(self):
        s = self.samples
        llm_iters = sum(
            1
            for sample in s
            for r in (sample.get("iteration_history") or [])
            if r.get("llm_call")
        )
        self.assertCell(
            "thesis_agent_behaviour",
            "LLM-driven iterations (total)",
            "Value",
            str(llm_iters),
            "Total LLM-driven iterations",
        )

        first = sum(1 for sample in s if sample.get("best_iteration") == 1)
        known = sum(1 for sample in s if isinstance(sample.get("best_iteration"), int))
        self.assertCell(
            "thesis_agent_behaviour",
            "Kept candidate came from iteration 1",
            "Value",
            f"{first}/{known} ({100.0 * first / known:.1f}\\%)",
            "Share of proteins whose kept candidate came from iteration 1",
        )

        changed_window = sum(
            1
            for sample in s
            for r in (sample.get("iteration_history") or [])
            if r.get("llm_call") and "junction_window" in (r.get("changed_levers") or {})
        )
        self.assertCell(
            "thesis_agent_behaviour",
            "Changed junction\\_window (share of LLM iterations)",
            "Value",
            f"{100.0 * changed_window / llm_iters:.1f}\\%",
            "Share of LLM iterations that changed junction_window",
        )

        modes = {}
        for sample in s:
            for r in sample.get("iteration_history") or []:
                value = (r.get("lever_values") or {}).get("search_mode")
                if value is not None:
                    modes[value] = modes.get(value, 0) + 1
        total = sum(modes.values())
        self.assertCell(
            "thesis_agent_behaviour",
            "search\\_mode values chosen",
            "Value",
            f"beam {100.0 * modes['beam'] / total:.1f}\\%, "
            f"greedy {100.0 * modes['greedy'] / total:.1f}\\%",
            "Share of iterations choosing each search mode",
        )

    # --- VII. error taxonomy ----------------------------------------------
    def test_error_taxonomy(self):
        """Failure classes rebuilt from the documented rules and cut points."""
        counts = {}
        for sample in self.samples:
            m = sample.get("recon_metrics") or {}
            em, apa = m.get("exact_match"), m.get("adjacent_pair_acc")
            lcr, tau = m.get("longest_correct_run"), m.get("kendall_tau")
            order = sample.get("order") or []
            if em is not None and em >= 1.0:
                label = "exact"
            elif any(
                v is None or (isinstance(v, float) and math.isnan(v))
                for v in (apa, lcr, tau)
            ):
                label = "unknown"
            else:
                n = len(order)
                bp = (n - 1) * (1.0 - apa) if n >= 2 else None
                if tau <= -0.5:
                    label = "block_reversal"
                elif abs(tau) < 0.15 and apa <= 0.05:
                    label = "full_scramble"
                elif bp is not None and bp <= 2 and lcr >= 0.5:
                    label = "local_transposition"
                elif order and order[0] != 0:
                    label = "wrong_start"
                else:
                    label = "partial_assembly"
            counts[label] = counts.get(label, 0) + 1

        total = sum(counts.values())
        self.assertCell(
            "thesis_error_taxonomy", "Exact reconstruction", "Proteins",
            str(counts["exact"]), "Count of exact reconstructions",
        )
        self.assertCell(
            "thesis_error_taxonomy", "Local transposition", "Proteins",
            str(counts["local_transposition"]), "Count of local transpositions",
        )
        self.assertCell(
            "thesis_error_taxonomy", "Exact reconstruction", "Share",
            f"{100.0 * counts['exact'] / total:.1f}\\%",
            "Share of exact reconstructions",
        )

    # --- VIII. cost --------------------------------------------------------
    def test_cost(self):
        s = self.samples
        n = len(s)
        calls = sum(x.get("llm_calls") or 0 for x in s)
        tokens = sum(x.get("llm_tokens") or 0 for x in s)
        self.assertCell(
            "thesis_cost", "LLM calls per protein", "Value", f"{calls / n:.2f}",
            "LLM calls per protein",
        )
        self.assertCell(
            "thesis_cost", "Total LLM calls / tokens", "Value", f"{calls} / {tokens}",
            "Total LLM calls and tokens",
        )
        completed = sum(1 for x in s if x.get("completed"))
        self.assertCell(
            "thesis_cost", "Completed reconstructions", "Value",
            f"{completed}/{n} ({100.0 * completed / n:.1f}\\%)",
            "Completed reconstructions",
        )
        agentic = sum(x.get("agentic_duration_seconds") or 0 for x in s) / n
        control = sum(x.get("control_duration_seconds") or 0 for x in s) / n
        self.assertCell(
            "thesis_cost", "Agentic / control time ratio", "Value",
            f"{agentic / control:.2f}", "Agentic / control wall-clock ratio",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
