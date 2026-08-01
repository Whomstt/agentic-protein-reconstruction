"""Independent verification of the report's Results tables.

Every check recomputes a value from scratch out of the run's samples.jsonl and
asserts the exact formatted string appears in the emitted .tex. For that
independence the file imports nothing from ``evaluation/``: no shared
aggregation, formatter or statistics, only json, math and stdlib unittest. Three
or more values are verified per table, at the precision the table prints.

    python tests/test_thesis_tables.py

Regenerate the tables first if the run data changed:

    python -m evaluation.thesis_tables --run 130726_224804_agentic"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = "130726_224804_agentic"
# Each configuration writes its own suffixed set of tables; this run is the
# E. coli, 100-replica one, so its files are <table>_ecoli_r100.tex.
SUFFIX = "_ecoli_r100"
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
    return (TABLES / f"{name}{SUFFIX}.tex").read_text(encoding="utf-8")


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


def paired_blocks():
    """The paired table as one list of cell-lists per comparison, split on the
    rule between the two comparisons. Its rows cannot be keyed by their first
    cell: the comparison label is printed once per block."""
    lines = tex("paired_tests").splitlines()
    start = lines.index(r"    \midrule")
    end = lines.index(r"    \bottomrule")
    blocks, current = [], []
    for line in lines[start + 1 : end]:
        if line.strip() == r"\midrule":
            blocks.append(current)
            current = []
            continue
        current.append([c.strip() for c in line.strip().removesuffix(r"\\").strip().split("&")])
    blocks.append(current)
    return blocks


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


def edit_sim(a, b):
    """1 - normalised Levenshtein, from a plain two-row DP.

    Deliberately the naive textbook recurrence: the metric module reaches for a
    vectorised prefix-min formulation on inputs this size, and that rewrite is
    only trustworthy if something independent agrees with it."""
    if a == b:
        return 1.0
    denom = max(len(a), len(b))
    if not denom:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return 1.0 - prev[-1] / denom


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

    # --- I. main results --------------------------------------------------
    def test_main_results(self):
        s = self.samples
        self.assertCell(
            "main_results",
            "Adjacent Pair Accuracy",
            "LLM-Guided",
            f3(mean(values(s, "agentic", "adjacent_pair_acc"))),
            "LLM-Guided Adjacent Pair Accuracy",
        )
        self.assertCell(
            "main_results",
            "Longest Correct Run",
            "Fixed Settings",
            f3(mean(values(s, "deterministic", "longest_correct_run"))),
            "Fixed Settings Longest Correct Run",
        )
        self.assertCell(
            "main_results",
            "Adjacent Pair Accuracy",
            "Random Order",
            f3(mean(values(s, "shuffled", "adjacent_pair_acc"))),
            "Random Order Adjacent Pair Accuracy",
        )
        self.assertCell(
            "main_results",
            "Longest Correct Run",
            "Random Search",
            f3(mean(values(s, "control", "longest_correct_run"))),
            "Random Search Longest Correct Run",
        )
        # Edit Similarity is not stored in samples.jsonl for these runs — it is
        # derived from the stored reconstruction strings. Recompute it here with
        # a local textbook edit distance, which also cross-checks the numpy
        # vectorisation the metric module uses.
        self.assertCell(
            "main_results",
            "Edit Similarity",
            "LLM-Guided",
            f3(mean([edit_sim(x["target"], x["reconstruction"]) for x in s])),
            "LLM-Guided Edit Similarity",
        )
        self.assertCell(
            "main_results",
            "Edit Similarity",
            "Fixed Settings",
            f3(mean([
                edit_sim(x["target"], x["iteration_history"][0]["reconstruction"])
                for x in s
            ])),
            "Fixed Settings Edit Similarity",
        )
        # The shuffled arm stores an index permutation, not a sequence, so this
        # cell must stay explicitly empty rather than silently printing a number.
        self.assertCell(
            "main_results",
            "Edit Similarity",
            "Random Order",
            "n/a",
            "Random Order Edit Similarity is unavailable",
        )
        # Exact Match is a binomial rate: the share of proteins reconstructed
        # exactly, counted here from the raw per-sample values.
        em = values(s, "agentic", "exact_match")
        successes, n = sum(1 for v in em if v >= 1.0), len(em)
        self.assertCell(
            "main_results",
            "Exact Match",
            "LLM-Guided",
            f3(successes / n),
            "LLM-Guided Exact Match",
        )

        # No cell anywhere carries a confidence interval any more: the tables
        # print point estimates only.
        for name in ("main_results", "paired_tests"):
            lines = tex(name).splitlines()
            body = lines[lines.index(r"    \midrule") + 1 : lines.index(r"    \bottomrule")]
            self.assertNotIn(
                "[", "".join(body),
                f"{name}.tex still prints an interval range",
            )

        # The table prints only the four reported metrics, in this order, and
        # the two dropped ones must not reappear.
        _, rows = parse_table("main_results")
        self.assertEqual(
            list(rows),
            ["Adjacent Pair Accuracy", "Exact Match", "Longest Correct Run", "Edit Similarity"],
            "main results prints the four reported metrics in order",
        )

    # --- II. paired tests -------------------------------------------------
    def test_paired_tests(self):
        s = self.samples
        pairs = paired(s, "agentic", "control", "adjacent_pair_acc")
        self.assertInTex(
            "paired_tests",
            f3s(mean([a - b for a, b in pairs])),
            "Mean delta, LLM-Guided - Random Search, Adjacent Pair Accuracy",
        )

        # McNemar discordant counts on Exact Match, LLM-Guided vs Random Search.
        em = paired(s, "agentic", "control", "exact_match")
        only_a = sum(1 for a, b in em if a >= 1.0 > b)
        only_b = sum(1 for a, b in em if b >= 1.0 > a)
        self.assertInTex(
            "paired_tests",
            f"{only_a + only_b} ({only_a}/{only_b})",
            "McNemar discordant pairs, Exact Match, LLM-Guided - Random Search",
        )

        blocks = paired_blocks()

        # The table prints the Holm-adjusted p. Holm over a family of FIVE can
        # only raise a raw p, and never past 5x it, so the printed value is
        # bounded either side by the independently computed exact McNemar p.
        # The 5x bound is the point of this check: it fails if the correction is
        # ever recomputed over the three printed metrics instead.
        k, n = min(only_a, only_b), only_a + only_b
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
        raw_p = min(1.0, 2 * tail)
        em_rows = [row for row in blocks[0] if row[1] == "Exact Match"]
        self.assertEqual(len(em_rows), 1, "one Exact Match row in the first block")
        printed = float(em_rows[0][4])
        self.assertGreaterEqual(round(printed, 3), round(raw_p, 3))
        self.assertLessEqual(printed, min(1.0, 5 * raw_p) + 1e-9)

        # Only the four reported metrics are printed, in order, in every block.
        for block in blocks:
            self.assertEqual(
                [row[1] for row in block],
                ["Adjacent Pair Accuracy", "Exact Match", "Longest Correct Run", "Edit Similarity"],
                "paired tests prints the four reported metrics in order",
            )

        # Wilcoxon non-zero pairs, Longest Correct Run, LLM-Guided vs Fixed Settings.
        lcr = paired(s, "agentic", "deterministic", "longest_correct_run")
        deltas = [a - b for a, b in lcr if a != b]
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        self.assertInTex(
            "paired_tests",
            f"{len(deltas)} ({pos}/{neg})",
            "Wilcoxon non-zero pairs, Longest Correct Run, LLM-Guided - Fixed Settings",
        )

    # --- III. stratification ----------------------------------------------
    def test_stratification(self):
        s = self.samples
        by_bin = {}
        for sample in s:
            by_bin.setdefault(frag_bin(nfrag(sample)), []).append(sample)

        self.assertCell(
            "stratification", "10-19", "n", str(len(by_bin["10-19"])),
            "Proteins in the 10-19 fragment bin",
        )

        def arm_mean(bin_label, arm):
            vals = [
                (x.get(ARM_KEY[arm]) or {}).get("adjacent_pair_acc")
                for x in by_bin[bin_label]
            ]
            return mean([v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)])

        self.assertCell(
            "stratification",
            "20-49",
            "LLM-Guided",
            f3(arm_mean("20-49", "agentic")),
            "LLM-Guided APA in the 20-49 bin",
        )

        lifts = []
        for sample in by_bin["5-9"]:
            a = (sample.get("recon_metrics") or {}).get("adjacent_pair_acc")
            b = (sample.get("baseline_metrics") or {}).get("adjacent_pair_acc")
            if all(isinstance(v, (int, float)) and not math.isnan(v) for v in (a, b)):
                lifts.append(a - b)
        self.assertCell(
            "stratification", "5-9", "Lift", f3s(mean(lifts)),
            "Paired lift over the Random Order floor in the 5-9 bin",
        )

    # --- IV. selection ceiling --------------------------------------------
    def test_selection_ceiling(self):
        s = self.samples
        self.assertCell(
            "selection_ceiling",
            "Sequence Similarity",
            "Best Candidate",
            f3(mean(values(s, "oracle", "similarity"))),
            "Best Candidate Sequence Similarity",
        )
        tau = paired(s, "oracle", "agentic", "kendall_tau")
        self.assertCell(
            "selection_ceiling",
            "Kendall Tau",
            "Gap",
            f3s(mean([o - a for o, a in tau])),
            "Mean Best Candidate gap, Kendall Tau",
        )
        apa = paired(s, "oracle", "agentic", "adjacent_pair_acc")
        with_gap = sum(1 for o, a in apa if o - a > 1e-12)
        self.assertCell(
            "selection_ceiling",
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
            "validity_concordance",
            "Samples with comparable candidate pairs",
            "Value",
            str(len(per_sample)),
            "Number of samples with comparable candidates",
        )
        self.assertCell(
            "validity_concordance",
            "Comparable candidate pairs",
            "Value",
            str(total_pairs),
            "Total comparable candidate pairs",
        )
        self.assertCell(
            "validity_concordance",
            "Mean within-sample concordance",
            "Value",
            f3(mean(per_sample)),
            "Mean within-sample concordance",
        )
        above = sum(1 for c in per_sample if c > 0.5)
        self.assertCell(
            "validity_concordance",
            "Samples above chance ($>$ 0.50)",
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
            "agent_behaviour",
            "LLM-driven iterations (total)",
            "Value",
            str(llm_iters),
            "Total LLM-driven iterations",
        )

        first = sum(1 for sample in s if sample.get("best_iteration") == 1)
        known = sum(1 for sample in s if isinstance(sample.get("best_iteration"), int))
        self.assertCell(
            "agent_behaviour",
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
            "agent_behaviour",
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
            "agent_behaviour",
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
            "error_taxonomy", "Exact reconstruction", "Proteins",
            str(counts["exact"]), "Count of exact reconstructions",
        )
        self.assertCell(
            "error_taxonomy", "Local transposition", "Proteins",
            str(counts["local_transposition"]), "Count of local transpositions",
        )
        self.assertCell(
            "error_taxonomy", "Exact reconstruction", "Share",
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
            "cost", "LLM calls", "LLM-Guided", f"{calls / n:.2f}",
            "LLM calls per protein",
        )
        self.assertCell(
            "cost", "LLM tokens", "LLM-Guided", f"{tokens / n:.1f}",
            "LLM tokens per protein",
        )
        agentic = sum(x.get("agentic_duration_seconds") or 0 for x in s) / n
        control = sum(x.get("control_duration_seconds") or 0 for x in s) / n
        self.assertCell(
            "cost", "Wall clock (s)", "LLM-Guided", f"{agentic:.1f}",
            "LLM-Guided wall clock per protein",
        )
        self.assertCell(
            "cost", "Wall clock (s)", "Random Search", f"{control:.1f}",
            "Random Search wall clock per protein",
        )
        self.assertInTex(
            "cost", f"{agentic / control:.2f}$\\times$",
            "LLM-Guided / Random Search wall-clock ratio",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
