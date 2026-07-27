"""Publication-format LaTeX tables for the report's Results section.

    python -m evaluation.thesis_tables --run 130726_224804_agentic

Every number is computed here from that run's ``samples.jsonl``; nothing is read
out of ``analysis_report.md``, ``report.md``, ``results.csv`` or ``summary.csv``,
and nothing is transcribed by hand. Each emitted ``.tex`` carries a provenance
comment naming the source run, the command that made it, the row count the
numbers were computed over, and the timestamp.

This module is a *composition* layer, not a second statistics implementation.
The aggregations come from ``evaluation/analysis.py``, the intervals and paired
tests from ``evaluation/rebuild.py`` (which is where they live for the
``analysis_report.md`` tables), and the rendering from ``evaluation/exports.py``.
So the thesis tables and the per-run analysis report cannot disagree: they are
the same numbers, formatted for a two-column IEEE float.

What is new here, and computed nowhere else, is Table VI: the agent's behaviour
across iterations (which levers it actually moved, and whether a later iteration
ever displaced the deterministic first pass), derived from the per-iteration
``lever_values`` / ``changed_levers`` / ``validity_score`` records that
``samples.jsonl`` already stores.

**No GPU, no model loading, no network.**
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

from evaluation.analysis import (
    RESULTS_ROOT,
    METRIC_KEYS,
    TAXONOMY_LABELS,
    TAXONOMY_ORDER,
    Run,
    arm_values,
    bin_labels,
    concordance_summary,
    cost_summary,
    exact_match_count,
    lift_over_baseline,
    load_run,
    oracle_gap,
    sample_rows,
    stratify_by_bin,
    taxonomy_counts,
)
from evaluation.exports import Table, fmt, fmt_ci, fmt_p, stamp_tables, write_tables_tex
from evaluation.metrics import METRIC_NAMES, nanmean

# The interval, paired-test and label machinery already used by the per-run
# analysis report. Imported, not reimplemented, so both outputs agree by
# construction.
from evaluation.rebuild import (
    BOOTSTRAP_SEED,
    CONFIDENCE,
    DEFAULT_RESAMPLES,
    PRIMARY_METRIC,
    _arm_label,
    metric_interval,
    paired_comparisons,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "report" / "tables"

# Paper precision. The analysis report prints 4 decimals; a printed table reads
# better at 3, and 3 is already past the resolution the CIs support.
PLACES = 3

LEVERS = ("junction_window", "search_mode", "beam_width", "edge_mode", "confirmed_bonus")

# Cells, headers and captions are written as PLAIN TEXT: exports.latex_escape
# escapes them on the way out (and maps Δ, τ, % and _ to their LaTeX forms), so
# hand-written markup here would be escaped into literal backslashes.


def _command(run_dir_name: str) -> str:
    return f"python -m evaluation.thesis_tables --run {run_dir_name}"


def _pct(numerator, denominator) -> str:
    if not denominator:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


# --------------------------------------------------------------------------
# Iteration-level derivations (used only by the agent-behaviour table)
# --------------------------------------------------------------------------


def iteration_behaviour(run: Run) -> dict:
    """What the agent actually did across its iteration budget.

    Reads ``iteration_history`` directly from the stored samples: per-iteration
    ``lever_values`` (the five values used), ``changed_levers`` (which of them
    the LLM moved relative to the previous iteration) and ``validity_score``
    (the selection signal that decides the kept candidate).

    'LLM iterations' are those with ``llm_call`` true — under
    ``run.iteration1_deterministic`` iteration 1 is a fixed-lever pass with no
    LLM call, so lever-change rates are reported over the LLM-driven iterations
    only, which is the denominator that means anything.
    """
    n_samples = 0
    total_iterations = 0
    llm_iterations = 0
    changed_counts = {lever: 0 for lever in LEVERS}
    value_counts: dict[str, dict] = {lever: {} for lever in LEVERS}
    distinct_combos = []
    best_iterations = []
    best_is_first = 0
    improved_on_first = 0
    first_pass_comparable = 0

    for sample in run.samples:
        history = sample.get("iteration_history") or []
        if not history:
            continue
        n_samples += 1
        total_iterations += len(history)

        combos = set()
        for record in history:
            levers = record.get("lever_values") or {}
            combos.add(tuple(str(levers.get(lever)) for lever in LEVERS))
            for lever in LEVERS:
                if lever in levers:
                    key = str(levers[lever])
                    value_counts[lever][key] = value_counts[lever].get(key, 0) + 1
            if record.get("llm_call"):
                llm_iterations += 1
                changed = record.get("changed_levers") or {}
                for lever in changed:
                    if lever in changed_counts:
                        changed_counts[lever] += 1
        distinct_combos.append(len(combos))

        best = sample.get("best_iteration")
        if isinstance(best, int):
            best_iterations.append(best)
            if best == 1:
                best_is_first += 1

        # Did any later iteration actually beat the first pass on the selection
        # signal? Lower validity is better.
        first = history[0].get("validity_score")
        later = [
            r.get("validity_score")
            for r in history[1:]
            if isinstance(r.get("validity_score"), (int, float))
        ]
        if isinstance(first, (int, float)) and later:
            first_pass_comparable += 1
            if min(later) < first:
                improved_on_first += 1

    return {
        "n_samples": n_samples,
        "total_iterations": total_iterations,
        "llm_iterations": llm_iterations,
        "mean_iterations": total_iterations / n_samples if n_samples else float("nan"),
        "mean_distinct_combos": nanmean(distinct_combos),
        "mean_best_iteration": nanmean(best_iterations),
        "best_is_first": best_is_first,
        "n_best_known": len(best_iterations),
        "improved_on_first": improved_on_first,
        "first_pass_comparable": first_pass_comparable,
        "changed_counts": changed_counts,
        "value_counts": value_counts,
    }


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def table_main_results(run: Run, rows, resamples) -> Table:
    """I - the headline table: every arm, every metric, with 95% CIs."""
    arms = run.arms
    table_rows = []
    for metric in METRIC_KEYS:
        cells = [METRIC_NAMES[metric]]
        for arm in arms:
            cells.append(fmt_ci(metric_interval(rows, arm, metric, resamples), PLACES))
        table_rows.append(cells)

    return Table(
        key="thesis_main_results",
        headers=["Metric"] + [_arm_label(run, a) for a in arms],
        rows=table_rows,
        caption=(
            f"Reconstruction quality on {run.organism} at {run.replica_count} digestion "
            f"replicas (n={len(rows)} proteins). Each cell is the mean with a 95% "
            "confidence interval: a Wilson score interval for Exact Match (a binomial "
            "count of successes out of n) and a BCa bootstrap for the four continuous "
            "metrics. The Shuffled Baseline is a random fragment ordering (a floor, not "
            "a method) and the Oracle picks, per metric, the best candidate the agent "
            "actually generated using ground truth (a ceiling, not a method)."
        ),
        label="tab:main_results",
        notes=(
            f"BCa bootstrap: {resamples} resamples, fixed seed {BOOTSTRAP_SEED}. Kendall "
            "tau ranges over [-1, 1]; every other metric over [0, 1]. Sequence Similarity "
            "is bought largely by fragment composition, which is identical across arms, "
            "so it should be read only as a delta against the shuffled floor."
        ),
        environment="table*",
        placement="!t",
    )


def table_paired_tests(run: Run, rows, comparisons) -> Table:
    """II - the significance table. Both paired comparisons in one float."""
    table_rows = []
    for label, entry in comparisons.items():
        comparison = entry["comparison"]
        for i, metric in enumerate(METRIC_KEYS):
            result = comparison["metrics"][metric]
            test, ci, holm = result["test"], result["delta_ci"], result["holm"]
            detail = test["detail"]
            if "discordant" in detail:
                test_name = "McNemar"
                pairs = (
                    f"{detail['discordant']} "
                    f"({detail['n10_only_a']}/{detail['n01_only_b']})"
                )
            else:
                test_name = "Wilcoxon"
                pairs = (
                    f"{detail['n_nonzero']} "
                    f"({detail['n_positive']}/{detail['n_negative']})"
                )
            table_rows.append(
                [
                    label if i == 0 else "",
                    METRIC_NAMES[metric],
                    fmt(ci["point"], PLACES, signed=True),
                    f"[{fmt(ci['low'], PLACES, signed=True)}, "
                    f"{fmt(ci['high'], PLACES, signed=True)}]",
                    test_name,
                    pairs,
                    fmt_p(test["pvalue"]),
                    fmt_p(holm["p_adjusted"]),
                    "yes" if holm["reject"] else "no",
                ]
            )

    return Table(
        key="thesis_paired_tests",
        headers=[
            "Comparison", "Metric", "Mean Δ", "95% CI", "Test",
            "Pairs (+/-)", "p", "p (Holm)", "Sig.",
        ],
        rows=table_rows,
        caption=(
            f"Paired per-sample comparisons on {run.organism} at r{run.replica_count} "
            f"(n={len(rows)} proteins). The arms run on the same proteins with the same "
            "iteration budget, tool pipeline and selection rule, so the tests are paired: "
            "an exact McNemar test on the discordant pairs for Exact Match, a Wilcoxon "
            "signed-rank test for the continuous metrics. Holm correction is applied "
            "across the five metrics within each comparison; alpha = 0.05."
        ),
        label="tab:paired_tests",
        notes=(
            "Pairs column: discordant pairs (Agentic-only/baseline-only) for McNemar, "
            "non-zero differences (positive/negative) for Wilcoxon. Mean Δ CIs are "
            "BCa bootstraps over per-sample differences, resampling whole proteins so "
            "the pairing is preserved. Agentic - Control isolates the LLM's reasoning "
            "from the value of trying several candidates and keeping the best; because "
            "iteration 1 is inside the agent's candidate set, Agentic - Deterministic "
            "cannot be negative on the selection signal and is the weaker claim."
        ),
        environment="table*",
        placement="!t",
    )


def table_stratification(run: Run, rows) -> Table:
    """III - difficulty stratification on the primary ordering metric."""
    arms = [a for a in run.arms if a != "oracle"]
    lift = lift_over_baseline(rows, PRIMARY_METRIC, "agentic")
    strat = {arm: stratify_by_bin(rows, arm, PRIMARY_METRIC) for arm in arms}

    table_rows = []
    for label in bin_labels():
        n_in_bin = strat["agentic"][label]["n"]
        if not n_in_bin:
            table_rows.append([label, "0"] + ["-"] * (len(arms) + 1))
            continue
        cells = [label, fmt(n_in_bin, 0)]
        for arm in arms:
            values = strat[arm][label]["values"]
            cells.append(fmt(nanmean(values), PLACES) if values else "-")
        cells.append(fmt(lift[label]["lift"], PLACES, signed=True))
        table_rows.append(cells)

    return Table(
        key="thesis_stratification",
        headers=["Fragments", "n"] + [_arm_label(run, a) for a in arms] + ["Lift"],
        rows=table_rows,
        caption=(
            f"{METRIC_NAMES[PRIMARY_METRIC]} by fragment count. Difficulty scales with "
            "how many pieces a protein was digested into: the number of possible "
            "orderings grows factorially while the evidence available at each junction "
            "does not. Lift is the mean paired per-sample difference between the Agentic "
            "arm and the shuffled floor within the bin, not a difference of two "
            "independent means."
        ),
        label="tab:stratification",
        notes=(
            "n counts proteins in the bin; a bin's mean is taken over those whose "
            "ordering metrics are defined (true fragment order recovered), so it can "
            "rest on fewer than n."
        ),
        environment="table*",
        placement="!t",
    )


def table_selection_ceiling(run: Run, rows) -> Table:
    """IV - what imperfect selection costs, per metric."""
    gaps = oracle_gap(rows)
    table_rows = [
        [
            METRIC_NAMES[metric],
            fmt(nanmean(arm_values(rows, "agentic", metric)), PLACES),
            fmt(nanmean(arm_values(rows, "oracle", metric)), PLACES),
            fmt(gaps[metric]["mean_gap"], PLACES, signed=True),
            f"{gaps[metric]['samples_with_gap']}/{gaps[metric]['n']}",
        ]
        for metric in METRIC_KEYS
    ]
    return Table(
        key="thesis_selection_ceiling",
        headers=["Metric", "Agentic", "Oracle", "Gap", "Samples with a gap"],
        rows=table_rows,
        caption=(
            "Selection ceiling. The Oracle takes, per metric, the best candidate the "
            "agent had already generated, using ground truth to choose. The gap is "
            "therefore quality the run reached and then discarded - recoverable by a "
            "better selection signal alone, with no additional search."
        ),
        label="tab:selection_ceiling",
        provenance={"n_rows": gaps[PRIMARY_METRIC]["n"], "row_unit": "samples with both arms defined"},
    )


def table_validity_concordance(run: Run, rows) -> Table:
    """V - is the selection signal trustworthy at all?"""
    concordance = concordance_summary(rows)
    search = run.config.get("search", {}) or {}
    table_rows = [
        ["Samples with comparable candidate pairs", fmt(concordance["n_samples"], 0)],
        ["Comparable candidate pairs", fmt(concordance["comparable_pairs"], 0)],
        ["Mean within-sample concordance", fmt(concordance["mean_concordance"], PLACES)],
        [
            "Samples above chance (> 0.50)",
            f"{concordance['above_chance']}/{concordance['n_samples']} "
            f"({_pct(concordance['above_chance'], concordance['n_samples'])})",
        ],
        ["Validity junction window", fmt(search.get("validity_junction_window"), 0)],
        ["Validity confirmed penalty", fmt(search.get("validity_confirmed_penalty"), 2)],
    ]
    return Table(
        key="thesis_validity_concordance",
        headers=["Measurement", "Value"],
        rows=table_rows,
        caption=(
            "Trust in the selection signal. Within each sample, concordance is the "
            "fraction of candidate pairs whose validity ordering (lower is better) "
            f"agrees with their true {METRIC_NAMES[PRIMARY_METRIC]} ordering, across the "
            "iterations that sample tried. 0.50 is a coin flip. Since the run keeps "
            "whichever candidate scores best on this signal, its concordance bounds what "
            "the search can deliver."
        ),
        label="tab:validity_concordance",
        provenance={
            "n_rows": concordance["n_samples"],
            "row_unit": "samples with >= 2 comparable candidates",
        },
    )


def table_agent_behaviour(run: Run, rows) -> Table:
    """VI - what the agent did with its five levers.

    The only table here not derivable from the per-sample metric blocks: it is
    computed from the per-iteration records inside ``samples.jsonl``.
    """
    b = iteration_behaviour(run)
    llm_iters = b["llm_iterations"]

    table_rows = [
        ["Iterations per protein (mean)", fmt(b["mean_iterations"], 2)],
        ["Distinct lever combinations tried per protein (mean)", fmt(b["mean_distinct_combos"], 2)],
        ["LLM-driven iterations (total)", fmt(llm_iters, 0)],
        ["Kept candidate came from iteration 1",
         f"{b['best_is_first']}/{b['n_best_known']} ({_pct(b['best_is_first'], b['n_best_known'])})"],
        ["Mean iteration of the kept candidate", fmt(b["mean_best_iteration"], 2)],
        ["A later iteration beat iteration 1 on validity",
         f"{b['improved_on_first']}/{b['first_pass_comparable']} "
         f"({_pct(b['improved_on_first'], b['first_pass_comparable'])})"],
    ]
    for lever in LEVERS:
        table_rows.append(
            [
                f"Changed {lever} (share of LLM iterations)",
                _pct(b["changed_counts"][lever], llm_iters),
            ]
        )
    for lever in ("search_mode", "edge_mode"):
        counts = b["value_counts"][lever]
        total = sum(counts.values())
        spread = ", ".join(
            f"{value} {_pct(count, total)}" for value, count in sorted(counts.items())
        )
        table_rows.append([f"{lever} values chosen", spread or "n/a"])

    return Table(
        key="thesis_agent_behaviour",
        headers=["Measurement", "Value"],
        rows=table_rows,
        caption=(
            "Agent behaviour across the iteration budget, computed from the per-iteration "
            "records. Iteration 1 runs the fixed default levers with no LLM call, so "
            "lever-change rates are taken over the LLM-driven iterations only. 'Kept "
            "candidate came from iteration 1' is the share of proteins on which no later "
            "attempt displaced that deterministic first pass."
        ),
        label="tab:agent_behaviour",
        notes=(
            "A lever counts as changed when it differs from the previous iteration's "
            "value. Percentages of LLM iterations use the total across all proteins as "
            "the denominator."
        ),
        provenance={"n_rows": b["n_samples"], "row_unit": "samples with iteration history"},
    )


def table_error_taxonomy(run: Run, rows) -> Table:
    """VII - the shape of the failures."""
    counts = taxonomy_counts(rows)
    total = sum(counts.values()) or 1
    table_rows = [
        [TAXONOMY_LABELS[key], fmt(counts[key], 0), _pct(counts[key], total)]
        for key in TAXONOMY_ORDER
        if counts.get(key)
    ]
    return Table(
        key="thesis_error_taxonomy",
        headers=["Failure mode", "Proteins", "Share"],
        rows=table_rows,
        caption=(
            "Error taxonomy of the Agentic arm's reconstructions. Each protein falls in "
            "exactly one class, checked most-specific first, and classified from the "
            "stored metric values (which were computed with the correct fragment-string "
            "semantics rather than recounted from fragment indices)."
        ),
        label="tab:error_taxonomy",
        notes=(
            "The cut points separating these classes are disclosed, untuned round "
            "numbers; they change only how a failure is labelled, never any headline "
            "metric."
        ),
    )


def table_cost(run: Run, rows) -> Table:
    """VIII - what the agentic arm costs against the control arm."""
    cost = cost_summary(rows)
    llm = run.config.get("llm_model", {}) or {}
    mlm = run.config.get("mlm_model", {}) or {}
    n = len(rows) or 1
    agentic_seconds = cost["agentic_seconds_per_sample"]
    control_seconds = cost["control_seconds_per_sample"]
    ratio = agentic_seconds / control_seconds if control_seconds else float("nan")

    table_rows = [
        ["LLM", str(llm.get("name", "n/a"))],
        ["Protein language model", str(mlm.get("name", "n/a"))],
        ["LLM calls per protein", fmt(cost["llm_calls_per_sample"], 2)],
        ["LLM tokens per protein", fmt(cost["llm_tokens_per_sample"], 1)],
        ["Total LLM calls / tokens",
         f"{fmt(cost['total_llm_calls'], 0)} / {fmt(cost['total_llm_tokens'], 0)}"],
        ["Lever-choice failures", fmt(cost["llm_failures"], 0)],
        ["Wall clock per protein, agentic arm", f"{fmt(agentic_seconds, 1)} s"],
        ["Wall clock per protein, control arm", f"{fmt(control_seconds, 1)} s"],
        ["Agentic / control time ratio", fmt(ratio, 2)],
        ["Completed reconstructions", f"{cost['completed']}/{n} ({_pct(cost['completed'], n)})"],
        ["True fragment order recovered",
         f"{cost['true_order_recovered']}/{n} ({_pct(cost['true_order_recovered'], n)})"],
    ]
    return Table(
        key="thesis_cost",
        headers=["Measurement", "Value"],
        rows=table_rows,
        caption=(
            "Cost, efficiency and completion. The control arm runs the same budget and "
            "pipeline with lever values from a non-LLM policy, so the time ratio is the "
            "price of the LLM's reasoning and the paired tests are the return on it. "
            "'True fragment order recovered' counts proteins whose fragments could be "
            "re-tiled against the target; the three ordering metrics are undefined on "
            "the remainder and those proteins are excluded from them rather than scored "
            "as zero."
        ),
        label="tab:cost",
    )


# The registry. Adding or removing a thesis table is one entry here.
BUILDERS = (
    ("thesis_main_results", lambda run, rows, ctx: table_main_results(run, rows, ctx["resamples"])),
    ("thesis_paired_tests", lambda run, rows, ctx: table_paired_tests(run, rows, ctx["comparisons"])),
    ("thesis_stratification", lambda run, rows, ctx: table_stratification(run, rows)),
    ("thesis_selection_ceiling", lambda run, rows, ctx: table_selection_ceiling(run, rows)),
    ("thesis_validity_concordance", lambda run, rows, ctx: table_validity_concordance(run, rows)),
    ("thesis_agent_behaviour", lambda run, rows, ctx: table_agent_behaviour(run, rows)),
    ("thesis_error_taxonomy", lambda run, rows, ctx: table_error_taxonomy(run, rows)),
    ("thesis_cost", lambda run, rows, ctx: table_cost(run, rows)),
)

# Tables that need an arm the run may not have produced.
REQUIRES_ARM = {
    "thesis_selection_ceiling": "oracle",
}


def build_tables(run_dir, out_dir: Path, resamples: int = DEFAULT_RESAMPLES,
                 quiet: bool = False) -> list[Path]:
    """Compute and write every thesis table for one run."""
    run = load_run(run_dir)
    rows = sample_rows(run)
    comparisons = paired_comparisons(run, rows)
    ctx = {"resamples": resamples, "comparisons": comparisons}

    tables: list[Table] = []
    skipped: list[str] = []
    for key, builder in BUILDERS:
        needed = REQUIRES_ARM.get(key)
        if needed and needed not in run.arms:
            skipped.append(f"{key} (run has no {needed} arm)")
            continue
        if key == "thesis_paired_tests" and not comparisons:
            skipped.append(f"{key} (run has no paired baseline arm)")
            continue
        tables.append(builder(run, rows, ctx))

    stamp_tables(
        tables,
        source_run=run.path.name,
        command=_command(run.path.name),
        n_rows=len(rows),
        source_file=f"results/{run.path.name}/samples.jsonl",
    )
    paths = write_tables_tex(tables, out_dir)

    if not quiet:
        print(f"{run.path.name} -> {out_dir}")
        for path in paths:
            print(f"  {path.name}")
        for note in skipped:
            print(f"  skipped: {note}")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.thesis_tables",
        description=(
            "Generate the report's Results tables as booktabs LaTeX, computed from a "
            "run's samples.jsonl. No GPU, no model loading, no network."
        ),
    )
    parser.add_argument("--run", required=True, help="run folder under results/ (or a path)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument(
        "--resamples", type=int, default=DEFAULT_RESAMPLES,
        help=f"bootstrap resamples (default {DEFAULT_RESAMPLES})",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.results_root)
    run_dir = Path(args.run) if Path(args.run).exists() else root / args.run
    if not (run_dir / "samples.jsonl").exists():
        print(f"No samples.jsonl under {run_dir}")
        return 1

    build_tables(run_dir, Path(args.out), resamples=args.resamples, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
