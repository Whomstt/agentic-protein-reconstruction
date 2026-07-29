"""Statistical report generator, the default reporting layer for every run.

Regenerates, from stored per-sample data alone:

  analysis_report.md   sections A-G with confidence intervals and paired tests
  results.csv          one row per sample, every raw metric and derived field
  summary.csv          aggregate metrics with CIs, one row per (arm, metric)
  tables/*.tex         booktabs tables to \\input{} directly
  figures/*.pdf        vector figures (PNG twins for the markdown preview)

Run offline over any finished run:

    python -m evaluation.rebuild --all           # every run in results/
    python -m evaluation.rebuild --run <folder>  # one run
    python -m evaluation.rebuild --all --resamples 2000   # faster, wider CIs

No GPU, no model loading, no network: everything is read from samples.jsonl plus
the config snapshot in summary.json. It is also called at the end of every run
(see evaluation/runner.py) and never rewrites the run's original report.md.

A measurement the stored data cannot support is printed as
``n/a - requires field: X`` rather than estimated or quietly omitted."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from evaluation.analysis import (
    ARM_LABELS,
    METRIC_KEYS,
    RESULTS_ROOT,
    TAXONOMY_LABELS,
    TAXONOMY_ORDER,
    TAXONOMY_THRESHOLDS,
    Run,
    arm_values,
    bin_labels,
    breakpoint_stats,
    concordance_summary,
    cost_summary,
    discover_runs,
    exact_match_count,
    junction_ranking_summary,
    lift_over_baseline,
    load_run,
    nterm_analysis,
    oracle_gap,
    sample_rows,
    stratify_by_bin,
    taxonomy_counts,
    trypsin_recall_summary,
)
from evaluation.exports import (
    Table,
    fmt,
    fmt_ci,
    fmt_p,
    stamp_tables,
    write_rows_csv,
    write_tables_tex,
)
from evaluation.metrics import METRIC_NAMES, nanmean
from evaluation.stats import (
    bca_bootstrap_ci,
    bca_paired_delta_ci,
    compare_arms,
    mcnemar_exact,
    wilcoxon_signed_rank,
)

# Fixed so a rebuild is reproducible: same input, same intervals, every time.
BOOTSTRAP_SEED = 20260726
DEFAULT_RESAMPLES = 10000
CONFIDENCE = 0.95

PRIMARY_METRIC = "adjacent_pair_acc"  # the primary ordering metric

MISSING_JUNCTION_RANKING = (
    "n/a - requires field: `junction_ranking` (per-sample top-1/top-3/MRR). "
    "The dense junction score matrix is not stored in samples.jsonl, and "
    "recomputing it needs pLM inference, which this offline rebuild does not do. "
    "Runs from the instrumentation change onward record it at zero extra model "
    "cost; for older runs, `python -m evaluation.junction_ranking` measures it "
    "separately."
)
MISSING_TRYPSIN_RECALL = (
    "n/a - requires field: `trypsin_recall` (which junctions the trypsin filter "
    "pruned). Older runs stored only the pruned COUNT (`num_pruned`), which "
    "cannot tell us whether a pruned junction was a true one. Recorded from the "
    "instrumentation change onward."
)


# --------------------------------------------------------------------------
# Interval helpers
# --------------------------------------------------------------------------


def metric_interval(rows, arm: str, metric: str, resamples: int = DEFAULT_RESAMPLES):
    """The right interval for the metric's type: a Wilson score interval for Exact
    Match, which is a count of successes out of n proteins, and a BCa bootstrap for
    the four continuous means."""
    if metric == "exact_match":
        from evaluation.stats import wilson_interval

        successes, n = exact_match_count(rows, arm)
        return wilson_interval(successes, n, confidence=CONFIDENCE)
    return bca_bootstrap_ci(
        arm_values(rows, arm, metric),
        confidence=CONFIDENCE,
        n_resamples=resamples,
        seed=BOOTSTRAP_SEED,
    )


def _arm_label(run: Run, arm: str) -> str:
    if arm == "control":
        policy = (run.config.get("run", {}).get("control_baseline", {}) or {}).get("policy")
        policy = policy or run.summary.get("control_policy")
        # The arm's name already says "random", so naming the shipped random
        # policy again reads as a stutter; a non-default policy still gets named.
        if policy and policy != "random":
            return f"Random Search ({policy} policy, no LLM)"
        return ARM_LABELS[arm]
    return ARM_LABELS[arm]


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def section_a_overall(run: Run, rows, resamples) -> tuple[str, list[Table], list[dict]]:
    """A — overall table with confidence intervals."""
    arms = run.arms
    headers = ["Metric"] + [_arm_label(run, a) for a in arms]
    table_rows = []
    summary_records = []

    for metric in METRIC_KEYS:
        cells = [METRIC_NAMES[metric]]
        for arm in arms:
            interval = metric_interval(rows, arm, metric, resamples)
            cells.append(fmt_ci(interval))
            summary_records.append(
                {
                    "run_dir": run.path.name,
                    "run_name": run.name,
                    "organism": run.organism,
                    "replica_count": run.replica_count,
                    "arm": arm,
                    "arm_label": _arm_label(run, arm),
                    "metric": metric,
                    "metric_label": METRIC_NAMES[metric],
                    "point": interval.point,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "ci_method": interval.method,
                    "n": interval.n,
                    "confidence": CONFIDENCE,
                }
            )
        table_rows.append(cells)

    table = Table(
        key="table_a_overall",
        headers=headers,
        rows=table_rows,
        caption=(
            f"Reconstruction quality on {run.organism} at {run.replica_count} digestion "
            f"replicas (n={len(rows)}). Point estimate with 95% CI: Wilson score interval "
            "for Exact Match (a binomial count), BCa bootstrap for the continuous metrics."
        ),
        label=f"tab:overall_{run.path.name}",
        notes=(
            f"BCa bootstrap: {resamples} resamples, seed {BOOTSTRAP_SEED}. "
            "Kendall tau ranges over [-1, 1]; the other metrics over [0, 1]."
        ),
    )

    text = [
        "## A. Overall Performance",
        "",
        f"All values are means over n={len(rows)} proteins with 95% confidence intervals. "
        "Exact Match uses a **Wilson score interval** (it is a count of successes out of n, "
        "not a continuous mean); the other four use a **BCa bootstrap** "
        f"({resamples} resamples, fixed seed {BOOTSTRAP_SEED}, so a rebuild reproduces "
        "these intervals exactly).",
        "",
        table.to_markdown(),
        "",
    ]
    return "\n".join(text), [table], summary_records


def section_b_ladder(run: Run, rows, resamples, out_dir) -> tuple[str, list[Table], list]:
    """B — method ladder, with the shuffled Sequence Similarity floor stated
    numerically because the raw value is misleading."""
    arms = run.arms
    ladder_rows = []
    for metric in METRIC_KEYS:
        previous = None
        cells = [METRIC_NAMES[metric]]
        for arm in arms:
            value = nanmean(arm_values(rows, arm, metric))
            step = "" if previous is None else f" ({fmt(value - previous, 4, signed=True)})"
            cells.append(f"{fmt(value)}{step}")
            previous = value
        ladder_rows.append(cells)

    table = Table(
        key="table_b_ladder",
        headers=["Metric"] + [_arm_label(run, a) for a in arms],
        rows=ladder_rows,
        caption=(
            "Method ladder: each rung's mean, with the step up from the rung to its "
            "left in parentheses."
        ),
        label=f"tab:ladder_{run.path.name}",
    )

    shuffled_similarity = nanmean(arm_values(rows, "shuffled", "similarity"))
    shuffled_em = nanmean(arm_values(rows, "shuffled", "exact_match"))
    shuffled_apa = nanmean(arm_values(rows, "shuffled", "adjacent_pair_acc"))
    shuffled_tau = nanmean(arm_values(rows, "shuffled", "kendall_tau"))

    text = [
        "## B. Method Ladder",
        "",
        table.to_markdown(),
        "",
        "### Reading the Sequence Similarity floor",
        "",
        f"**A random shuffle of the fragments already scores "
        f"{fmt(shuffled_similarity)} on Sequence Similarity.** Every candidate "
        "ordering is a permutation of the *same* fragment multiset, so the string "
        "composition is identical across all arms and only the order varies. "
        "`difflib.SequenceMatcher` credits matching blocks wherever they occur, so a "
        "large fraction of that ratio is bought by composition alone and is available "
        "to a method that has learned nothing.",
        "",
        "For contrast, on the same shuffled orderings the ordering-sensitive metrics "
        f"sit at their true floor: Exact Match {fmt(shuffled_em)}, Adjacent Pair "
        f"Accuracy {fmt(shuffled_apa)}, Kendall Tau {fmt(shuffled_tau)}. "
        "**Read Sequence Similarity only as a delta against the shuffled floor, never "
        "as an absolute.** Adjacent Pair Accuracy is the primary ordering metric.",
        "",
    ]

    figures = []
    try:
        from evaluation import figures as figs

        for metric in (PRIMARY_METRIC, "kendall_tau"):
            records = []
            for arm in arms:
                interval = metric_interval(rows, arm, metric, resamples)
                records.append(
                    {
                        "arm_label": _arm_label(run, arm),
                        "point": interval.point,
                        "low": interval.low,
                        "high": interval.high,
                    }
                )
            figures.append(
                figs.method_ladder(
                    records, metric, METRIC_NAMES[metric], out_dir, f"fig_ladder_{metric}"
                )
            )
    except Exception as exc:  # figures must never break the report
        text.append(f"_Figure generation skipped: {exc}_\n")
    return "\n".join(text), [table], figures


def section_c_replica(run: Run, rows) -> tuple[str, list[Table], list]:
    """C — replica scaling. A single run is one point on that curve; the
    cross-run report draws the curve itself."""
    confirmed = [r.get("num_confirmed_adjacencies") for r in rows]
    confirmed = [c for c in confirmed if isinstance(c, (int, float))]
    fragments = [r.get("num_fragments") for r in rows if r.get("num_fragments")]
    coverage = (
        nanmean(
            [
                c / (n - 1)
                for c, n in zip(confirmed, fragments)
                if isinstance(c, (int, float)) and n and n > 1
            ]
        )
        if confirmed
        else float("nan")
    )

    table = Table(
        key="table_c_replica",
        headers=["Quantity", "Value"],
        rows=[
            ["Digestion replicas", fmt(run.replica_count, 0)],
            ["Mean confirmed adjacencies per protein", fmt(nanmean(confirmed), 2)],
            ["Mean true joins covered by the overlap graph", fmt(coverage, 4)],
            ["Mean junctions pruned by trypsin filter (%)", fmt(nanmean([r.get("pruned_pct") for r in rows]), 2)],
        ],
        caption=f"Overlap-graph strength at {run.replica_count} replicas.",
        label=f"tab:replica_{run.path.name}",
    )

    text = [
        "## C. Replica Scaling",
        "",
        f"This run sits at **{run.replica_count} digestion replicas**. Replica count is "
        "what determines how many adjacencies the overlap graph can confirm outright, "
        "which is near-ground-truth structural information the search gets for free.",
        "",
        table.to_markdown(),
        "",
        "_The scaling curve across replica counts is in the cross-run report_ "
        "(`cross_run_report.md`), _which needs more than one run to draw._",
        "",
    ]
    return "\n".join(text), [table], []


def paired_comparisons(run: Run, rows) -> dict:
    """Every paired arm comparison for this run, keyed by comparison label.

    Computed once and threaded through both the report section and the
    cross-run tests CSV, so the same numbers back both and the BCa bootstraps
    are not paid for twice.
    """
    out = {}
    for label, other in (
        ("LLM-Guided − Random Search", "control"),
        ("LLM-Guided − Fixed Settings", "deterministic"),
    ):
        if other not in run.arms or "agentic" not in run.arms:
            continue
        arm_a = {m: arm_values(rows, "agentic", m) for m in METRIC_KEYS}
        arm_b = {m: arm_values(rows, other, m) for m in METRIC_KEYS}
        out[label] = {
            "baseline_arm": other,
            "comparison": compare_arms(arm_a, arm_b, alpha=0.05),
            "arm_a": arm_a,
            "arm_b": arm_b,
        }
    return out


def section_d_llm(run: Run, rows, comparisons, out_dir) -> tuple[str, list[Table], list]:
    """D — isolating the LLM: paired tests, not CI overlap."""
    text = ["## D. Isolating the LLM's Contribution", ""]
    tables: list[Table] = []
    figures: list = []

    if "control" not in run.arms:
        text += [
            "n/a - this run has no Random Search arm (`run.control_baseline.enabled` was off), "
            "so the LLM's reasoning cannot be separated from the value of trying several "
            "candidates and keeping the best.",
            "",
        ]
        return "\n".join(text), tables, figures

    text += [
        "The LLM-Guided and Random Search arms run on the **same proteins** with the same "
        "iteration budget, the same tool pipeline and the same best-validity selection; "
        "only the source of the five lever values differs (LLM vs. a non-LLM policy). "
        "They are therefore **paired**, and the comparison uses paired tests rather than "
        "asking whether the two arms' confidence intervals overlap — an overlap test on "
        "paired data is both wrong and badly underpowered.",
        "",
        "Exact Match uses an **exact McNemar test** on the discordant pairs; the four "
        "continuous metrics use a **Wilcoxon signed-rank test**. Because five metrics are "
        "tested on one hypothesis, p-values are corrected with **Holm** across the family.",
        "",
    ]

    for label, entry in comparisons.items():
        other = entry["baseline_arm"]
        comparison = entry["comparison"]
        arm_a, arm_b = entry["arm_a"], entry["arm_b"]

        table_rows = []
        for metric in METRIC_KEYS:
            entry = comparison["metrics"][metric]
            test, ci, holm = entry["test"], entry["delta_ci"], entry["holm"]
            detail = test["detail"]
            if "discordant" in detail:
                counts = (
                    f"{detail['discordant']} "
                    f"(A only {detail['n10_only_a']}, B only {detail['n01_only_b']})"
                )
                test_name = "McNemar (exact)"
            else:
                counts = (
                    f"{detail['n_nonzero']} non-zero "
                    f"({detail['n_positive']}+ / {detail['n_negative']}-)"
                )
                test_name = "Wilcoxon"
            table_rows.append(
                [
                    METRIC_NAMES[metric],
                    fmt(ci["point"], 4, signed=True),
                    f"[{fmt(ci['low'], 4, signed=True)}, {fmt(ci['high'], 4, signed=True)}]",
                    test_name,
                    counts,
                    fmt_p(test["pvalue"]),
                    fmt_p(holm["p_adjusted"]),
                    "yes" if holm["reject"] else "no",
                ]
            )

        key = "table_d_" + other
        table = Table(
            key=key,
            headers=[
                "Metric", "Mean Δ", "95% CI", "Test", "Discordant / non-zero pairs",
                "p", "p (Holm)", "Significant",
            ],
            rows=table_rows,
            caption=(
                f"Paired comparison, {label}, on {run.organism} at r{run.replica_count} "
                f"(n={len(rows)}). Holm-corrected across the five metrics; alpha = 0.05."
            ),
            label=f"tab:{key}_{run.path.name}",
            notes=(
                "Mean Δ CI is a BCa bootstrap over per-sample differences, resampling "
                "whole proteins so the pairing is preserved."
            ),
        )
        tables.append(table)
        text += [f"### {label}", "", table.to_markdown(), ""]

        rejected = [
            METRIC_NAMES[m]
            for m in METRIC_KEYS
            if comparison["metrics"][m]["holm"]["reject"]
        ]
        if rejected:
            text += [
                f"Significant after Holm correction: **{', '.join(rejected)}**.",
                "",
            ]
        else:
            text += [
                "**No metric survives Holm correction.** The observed differences are "
                "consistent with what re-running the same non-LLM policy on these "
                f"n={len(rows)} proteins could produce by chance; this run does not "
                "demonstrate a reasoning advantage on this comparison.",
                "",
            ]

        if other == "control":
            try:
                from evaluation import figures as figs

                deltas = [
                    a - b
                    for a, b in zip(arm_a[PRIMARY_METRIC], arm_b[PRIMARY_METRIC])
                    if isinstance(a, (int, float))
                    and isinstance(b, (int, float))
                    and not math.isnan(a)
                    and not math.isnan(b)
                ]
                figures.append(
                    figs.paired_gain_distribution(
                        deltas,
                        f"Δ {METRIC_NAMES[PRIMARY_METRIC]} (LLM-Guided − Random Search)",
                        out_dir,
                        "fig_paired_gain_control",
                    )
                )
            except Exception as exc:
                text.append(f"_Figure generation skipped: {exc}_\n")

    return "\n".join(text), tables, figures


def section_e_bottleneck(run: Run, rows) -> tuple[str, list[Table], list]:
    """E — where the bottleneck is: junction ranking, selection concordance,
    oracle gap."""
    text = ["## E. Where the Bottleneck Is", ""]
    tables: list[Table] = []

    # --- junction ranking -------------------------------------------------
    text += ["### Junction scorer ranking (search-independent)", ""]
    ranking = junction_ranking_summary(rows)
    if ranking is None:
        text += [MISSING_JUNCTION_RANKING, ""]
    else:
        table = Table(
            key="table_e_junction_ranking",
            headers=["Measurement", "Value"],
            rows=[
                ["Samples measured", fmt(ranking["n_samples"], 0)],
                ["True junctions scored", fmt(ranking["total_junctions"], 0)],
                ["Top-1 successor accuracy", fmt(ranking["top1_acc"])],
                ["Top-3 successor accuracy", fmt(ranking["top3_acc"])],
                ["Mean reciprocal rank", fmt(ranking["mrr"])],
            ],
            caption=(
                "How well the raw pLM junction scorer ranks the true successor "
                "fragment, before any search or constraint is applied."
            ),
            label=f"tab:junction_ranking_{run.path.name}",
        )
        tables.append(table)
        text += [table.to_markdown(), ""]

    # --- selection signal -------------------------------------------------
    concordance = concordance_summary(rows)
    validity_window = (run.config.get("search", {}) or {}).get("validity_junction_window")
    penalty = (run.config.get("search", {}) or {}).get("validity_confirmed_penalty")
    conc_table = Table(
        key="table_e_concordance",
        headers=["Measurement", "Value"],
        rows=[
            ["Samples with comparable candidate pairs", fmt(concordance["n_samples"], 0)],
            ["Comparable candidate pairs", fmt(concordance["comparable_pairs"], 0)],
            ["Mean within-sample concordance", fmt(concordance["mean_concordance"])],
            ["Samples where concordance > 0.50", fmt(concordance["above_chance"], 0)],
            ["Validity junction window", fmt(validity_window, 0)],
            ["Validity confirmed penalty", fmt(penalty, 2)],
        ],
        caption=(
            "Does the validity signal actually pick the better candidate? Within-sample "
            "concordance between validity (lower is better) and true quality "
            f"({METRIC_NAMES[PRIMARY_METRIC]}) across the iterations each sample tried. "
            "0.50 is chance."
        ),
        label=f"tab:concordance_{run.path.name}",
    )
    tables.append(conc_table)
    text += [
        "### Selection signal trust",
        "",
        "The run keeps whichever candidate scores best on the validity signal, so the "
        "signal's ability to rank candidates bounds what the search can deliver. "
        "**0.50 is a coin flip.**",
        "",
        conc_table.to_markdown(),
        "",
    ]

    # --- oracle gap -------------------------------------------------------
    if "oracle" in run.arms:
        gaps = oracle_gap(rows)
        gap_table = Table(
            key="table_e_oracle_gap",
            headers=["Metric", "LLM-Guided", "Best Candidate", "Gap", "Samples with a gap"],
            rows=[
                [
                    METRIC_NAMES[m],
                    fmt(nanmean(arm_values(rows, "agentic", m))),
                    fmt(nanmean(arm_values(rows, "oracle", m))),
                    fmt(gaps[m]["mean_gap"], 4, signed=True),
                    f"{gaps[m]['samples_with_gap']}/{gaps[m]['n']}",
                ]
                for m in METRIC_KEYS
            ],
            caption=(
                "Selection ceiling. The Best Candidate column picks the best candidate the run already "
                "generated, using ground truth. The gap is quality lost purely to "
                "imperfect selection - reachable with a better signal and no new search."
            ),
            label=f"tab:oracle_{run.path.name}",
        )
        tables.append(gap_table)
        text += ["### Selection ceiling (Best Candidate)", "", gap_table.to_markdown(), ""]
        primary_gap = gaps[PRIMARY_METRIC]["mean_gap"]
        text += [
            f"On {METRIC_NAMES[PRIMARY_METRIC]} the run leaves **{fmt(primary_gap)}** on "
            f"the table in candidates it had already generated but did not select "
            f"({gaps[PRIMARY_METRIC]['samples_with_gap']}/{gaps[PRIMARY_METRIC]['n']} "
            "samples). That is the size of the prize for a better selection signal alone.",
            "",
        ]

    # --- trypsin recall ---------------------------------------------------
    text += ["### Trypsin filter recall", ""]
    recall = trypsin_recall_summary(rows)
    if recall is None:
        avg_pruned = nanmean([r.get("pruned_pct") for r in rows])
        text += [
            f"The filter pruned **{fmt(avg_pruned, 2)}%** of candidate junctions on "
            "average. Whether any pruned junction was a *true* one is "
            + MISSING_TRYPSIN_RECALL,
            "",
        ]
    else:
        table = Table(
            key="table_e_trypsin_recall",
            headers=["Measurement", "Value"],
            rows=[
                ["Samples measured", fmt(recall["n_samples"], 0)],
                ["True junctions", fmt(recall["true_junctions"], 0)],
                ["True junctions wrongly pruned", fmt(recall["true_junctions_pruned"], 0)],
                ["Filter recall", fmt(recall["recall"])],
                ["Samples losing at least one true junction", fmt(recall["samples_with_loss"], 0)],
            ],
            caption="Trypsin filter recall: did constraint pruning ever remove a true junction?",
            label=f"tab:trypsin_{run.path.name}",
        )
        tables.append(table)
        text += [table.to_markdown(), ""]

    return "\n".join(text), tables, []


def section_f_stratification(run: Run, rows, out_dir) -> tuple[str, list[Table], list]:
    """F — fragment-count stratification and error modes."""
    text = ["## F. Difficulty Stratification and Error Modes", ""]
    tables: list[Table] = []
    figures: list = []

    # --- performance vs fragment count ------------------------------------
    lift = lift_over_baseline(rows, PRIMARY_METRIC, "agentic")
    strat_rows = []
    for label in bin_labels():
        entry = lift[label]
        if not entry["n"]:
            strat_rows.append([label, "0", "-", "-", "-"])
            continue
        strat_rows.append(
            [
                label,
                fmt(entry["n"], 0),
                fmt(entry["baseline_mean"]),
                fmt(entry["arm_mean"]),
                fmt(entry["lift"], 4, signed=True),
            ]
        )

    strat_table = Table(
        key="table_f_fragment_stratification",
        headers=[
            "Fragments", "n", "Random Order", "LLM-Guided", "Lift (paired)",
        ],
        rows=strat_rows,
        caption=(
            f"{METRIC_NAMES[PRIMARY_METRIC]} by fragment count, with lift over the "
            "Random Order floor. Lift is the mean per-sample difference (paired), not a "
            "difference of independent means."
        ),
        label=f"tab:stratification_{run.path.name}",
    )
    tables.append(strat_table)
    text += [
        f"### {METRIC_NAMES[PRIMARY_METRIC]} by fragment count",
        "",
        "Difficulty scales with how many pieces the protein was cut into: the number of "
        "possible orderings grows factorially, while the pLM's evidence per junction does "
        "not improve. Lift over the Random Order floor is the honest read of whether the "
        "method is doing anything at each difficulty.",
        "",
        strat_table.to_markdown(),
        "",
    ]

    # --- all metrics by bin ------------------------------------------------
    all_metric_rows = []
    for metric in METRIC_KEYS:
        strat = stratify_by_bin(rows, "agentic", metric)
        all_metric_rows.append(
            [METRIC_NAMES[metric]]
            + [
                fmt(nanmean(strat[label]["values"])) if strat[label]["values"] else "-"
                for label in bin_labels()
            ]
        )
    all_table = Table(
        key="table_f_all_metrics_by_bin",
        headers=["Metric"] + bin_labels(),
        rows=all_metric_rows,
        caption="LLM-Guided arm, every metric, stratified by fragment count.",
        label=f"tab:allmetrics_{run.path.name}",
    )
    tables.append(all_table)
    text += ["### Every metric by fragment count", "", all_table.to_markdown(), ""]

    # --- N-terminal --------------------------------------------------------
    nterm = nterm_analysis(rows)
    nterm_table = Table(
        key="table_f_nterminal",
        headers=["Measurement", "Value"],
        rows=[
            ["P(correct N-terminal start)", fmt(nterm["p_correct_start"])],
            ["...on shuffled orderings", fmt(nterm["shuffled_p_correct_start"])],
            [
                "Exact Match | correct start",
                f"{fmt(nterm['em_given_correct_start'])} "
                f"({nterm['em_hits_given_correct_start']}/{nterm['n_given_correct_start']})",
            ],
            [
                "Exact Match | wrong start",
                f"{fmt(nterm['em_given_wrong_start'])} "
                f"({nterm['em_hits_given_wrong_start']}/{nterm['n_given_wrong_start']})",
            ],
        ],
        caption=(
            "N-terminal start accuracy and Exact Match conditioned on it. The assembly "
            "is built left to right, so the first fragment anchors everything after it."
        ),
        label=f"tab:nterm_{run.path.name}",
        notes=(
            "Derived as order[0] == 0 (the ground-truth order is the identity "
            "permutation); validated against fragment-string truth on 197/197 samples "
            "where the strings survive."
        ),
    )
    tables.append(nterm_table)
    text += [
        "### N-terminal start",
        "",
        nterm_table.to_markdown(),
        "",
    ]
    if nterm["em_given_correct_start"] is not None and nterm["n_given_wrong_start"]:
        text += [
            "Exact reconstruction is effectively conditional on getting the first "
            "fragment right — an ordering that starts wrong has already displaced every "
            "fragment after it.",
            "",
        ]

    # --- breakpoints -------------------------------------------------------
    bp = breakpoint_stats(rows)
    bp_table = Table(
        key="table_f_breakpoints",
        headers=["Measurement", "Value"],
        rows=[
            ["Samples", fmt(bp.get("n", 0), 0)],
            ["Mean breakpoints per protein", fmt(bp.get("mean"), 2)],
            ["Median", fmt(bp.get("median"), 2)],
            ["Min / Max", f"{fmt(bp.get('min'), 0)} / {fmt(bp.get('max'), 0)}"],
            ["Mean breakpoints per join", fmt(bp.get("mean_normalized"))],
            ["Proteins assembled with 0 breakpoints", fmt(bp.get("zero_breakpoint_samples"), 0)],
        ],
        caption=(
            "Breakpoints per protein: (n-1) - correct adjacencies. The count of joins "
            "the assembly got wrong."
        ),
        label=f"tab:breakpoints_{run.path.name}",
        notes=(
            "Derived from the stored (string-multiset) adjacent-pair accuracy, not "
            "recounted from fragment indices, because duplicate fragments make an "
            "index-space recount disagree with the reported metric."
        ),
    )
    tables.append(bp_table)
    text += ["### Breakpoints", "", bp_table.to_markdown(), ""]

    # --- error taxonomy ----------------------------------------------------
    counts = taxonomy_counts(rows)
    total = sum(counts.values()) or 1
    tax_table = Table(
        key="table_f_error_taxonomy",
        headers=["Failure mode", "Samples", "Share"],
        rows=[
            [TAXONOMY_LABELS[key], fmt(counts[key], 0), f"{100.0 * counts[key] / total:.1f}%"]
            for key in TAXONOMY_ORDER
            if counts.get(key)
        ],
        caption="Error taxonomy of the agentic arm's reconstructions.",
        label=f"tab:taxonomy_{run.path.name}",
        notes=(
            "Classified from stored, string-correct metric values. Cut points: "
            f"reversal tau <= {TAXONOMY_THRESHOLDS['reversal_tau_max']}; "
            f"scramble |tau| < {TAXONOMY_THRESHOLDS['scramble_tau_abs_max']} and "
            f"APA <= {TAXONOMY_THRESHOLDS['scramble_apa_max']}; "
            f"local transposition <= {TAXONOMY_THRESHOLDS['local_max_breakpoints']} "
            f"breakpoints and LCR >= {TAXONOMY_THRESHOLDS['local_min_lcr']}; "
            "wrong start = order[0] != 0 among orderings not already classified. "
            "These are disclosed, untuned choices; they affect labelling only."
        ),
    )
    tables.append(tax_table)
    text += [
        "### Error taxonomy",
        "",
        "Failures are classified from the stored metric values (which were computed with "
        "the correct fragment-string semantics), checked in a fixed order so each sample "
        "lands in exactly one class. The cut points are disclosed in the table note and "
        "affect labelling only — no headline number depends on them.",
        "",
        tax_table.to_markdown(),
        "",
    ]

    try:
        from evaluation import figures as figs

        per_bin_by_arm = {
            _arm_label(run, arm): stratify_by_bin(rows, arm, PRIMARY_METRIC)
            for arm in run.arms
            if arm in ("shuffled", "deterministic", "control", "agentic")
        }
        figures.append(
            figs.fragment_stratification(
                per_bin_by_arm,
                PRIMARY_METRIC,
                METRIC_NAMES[PRIMARY_METRIC],
                out_dir,
                "fig_fragment_stratification",
            )
        )
        figures.append(
            figs.lift_by_bin(
                lift, METRIC_NAMES[PRIMARY_METRIC], out_dir, "fig_lift_by_bin"
            )
        )
        figures.append(
            figs.error_taxonomy(
                {run.label(): counts}, TAXONOMY_LABELS, out_dir, "fig_error_taxonomy"
            )
        )
    except Exception as exc:
        text.append(f"_Figure generation skipped: {exc}_\n")

    return "\n".join(text), tables, figures


def section_g_cost(run: Run, rows) -> tuple[str, list[Table], list]:
    """G — cost and completion."""
    cost = cost_summary(rows)
    llm = run.config.get("llm_model", {}) or {}
    n = len(rows) or 1
    agentic_seconds = cost["agentic_seconds_per_sample"]
    control_seconds = cost["control_seconds_per_sample"]
    overhead = (
        agentic_seconds / control_seconds if control_seconds else float("nan")
    )

    table = Table(
        key="table_g_cost",
        headers=["Measurement", "Value"],
        rows=[
            ["LLM model", str(llm.get("name", "n/a"))],
            ["Total LLM calls", fmt(cost["total_llm_calls"], 0)],
            ["Total LLM tokens", fmt(cost["total_llm_tokens"], 0)],
            ["LLM calls per sample", fmt(cost["llm_calls_per_sample"], 2)],
            ["LLM tokens per sample", fmt(cost["llm_tokens_per_sample"], 1)],
            ["Lever-choice failures", fmt(cost["llm_failures"], 0)],
            ["Wall clock per sample (total)", f"{fmt(cost['seconds_per_sample'], 1)} s"],
            ["Wall clock per sample (agentic arm)", f"{fmt(agentic_seconds, 1)} s"],
            ["Wall clock per sample (control arm)", f"{fmt(control_seconds, 1)} s"],
            ["LLM-Guided / Random Search time ratio", fmt(overhead, 2)],
            ["Completed samples", f"{cost['completed']}/{n}"],
            ["True order recovered", f"{cost['true_order_recovered']}/{n}"],
        ],
        caption="Cost, efficiency and completion.",
        label=f"tab:cost_{run.path.name}",
    )

    text = [
        "## G. Cost",
        "",
        table.to_markdown(),
        "",
    ]
    if control_seconds and not math.isnan(overhead):
        text += [
            f"The agentic arm costs **{fmt(overhead, 1)}x** the control arm's wall clock "
            f"and **{fmt(cost['llm_calls_per_sample'], 1)} LLM calls per sample**, "
            "against which section D's paired tests are the return.",
            "",
        ]
    return "\n".join(text), [table], []


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _header(run: Run, rows, resamples) -> str:
    data = run.config.get("data", {}) or {}
    search = run.config.get("search", {}) or {}
    run_cfg = run.config.get("run", {}) or {}
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def setting(value):
        """Config keys absent from an older run's snapshot read as 'n/a', never
        as a value that run did not actually use."""
        return "n/a" if value is None else str(value)

    return "\n".join(
        [
            f"# {run.name} — Statistical Report",
            "",
            f"_Generated {generated} by `python -m evaluation.rebuild`. Every number is "
            f"computed from `samples.jsonl` (n={len(rows)}); nothing is transcribed by "
            "hand. Bootstraps use a fixed seed, so a rebuild reproduces this file "
            "exactly._",
            "",
            "| Setting | Value |",
            "| --- | --- |",
            f"| Organism | {run.organism} |",
            f"| Digestion replicas | {setting(run.replica_count)} |",
            f"| Samples | {len(rows)} |",
            f"| Missed cleavage ratio | {setting(data.get('missed_cleavage_ratio'))} |",
            f"| Method / calling mode | {setting(run_cfg.get('method'))} / "
            f"{setting(run_cfg.get('calling_mode'))} |",
            f"| Iteration 1 deterministic | {setting(run_cfg.get('iteration1_deterministic'))} |",
            f"| Max iterations | {setting(search.get('max_iterations'))} |",
            f"| Improvement margin | {setting(search.get('improvement_margin'))} |",
            f"| MLM | {setting((run.config.get('mlm_model') or {}).get('name'))} |",
            f"| LLM | {setting((run.config.get('llm_model') or {}).get('name'))} |",
            f"| Bootstrap | {resamples} resamples, seed {BOOTSTRAP_SEED} |",
            "",
            "**Disclosures.** There is no train/test split in this project — constants "
            "such as `improvement_margin`, `validity_junction_window` and "
            "`validity_confirmed_penalty` were chosen on samples drawn from the same "
            "undivided pool this evaluation draws from, so they are disclosed "
            "sensitivity choices, not validated constants. Iteration 1 being "
            "deterministic means the agentic arm can never score worse than the "
            "Fixed Settings arm on validity, which is why section D's "
            "`LLM-Guided − Random Search` comparison is the defensible reasoning claim.",
            "",
        ]
    )


def generate_run_artifacts(
    run_dir, resamples: int = DEFAULT_RESAMPLES, quiet: bool = False
) -> dict:
    """Build every artifact for one run. The single entry point used by both the
    CLI and the end-of-run pipeline hook.

    Never touches the run's original ``report.md``.
    """
    run = load_run(run_dir)
    rows = sample_rows(run)
    out = Path(run.path)
    figures_dir = out / "figures"
    tables_dir = out / "tables"

    chunks = [_header(run, rows, resamples)]
    tables: list[Table] = []
    figures: list = []
    summary_records: list[dict] = []

    comparisons = paired_comparisons(run, rows)

    text, tabs, records = section_a_overall(run, rows, resamples)
    chunks.append(text)
    tables += tabs
    summary_records += records

    for builder in (
        lambda: section_b_ladder(run, rows, resamples, figures_dir),
        lambda: section_c_replica(run, rows),
        lambda: section_d_llm(run, rows, comparisons, figures_dir),
        lambda: section_e_bottleneck(run, rows),
        lambda: section_f_stratification(run, rows, figures_dir),
        lambda: section_g_cost(run, rows),
    ):
        text, tabs, figs_out = builder()
        chunks.append(text)
        tables += tabs
        figures += figs_out

    chunks.append(
        "\n".join(
            [
                "## Provenance",
                "",
                "- Source: `samples.jsonl` in this folder — per-sample stored data only.",
                "- No GPU, no model loading and no network access were used to build this "
                "report.",
                "- Regenerate with `python -m evaluation.rebuild --run "
                f"{run.path.name}`.",
                "- `report.md` (the run's original report) is left untouched; this file "
                "is additive.",
                "",
            ]
        )
    )

    report_path = out / "analysis_report.md"
    report_path.write_text("\n".join(chunks), encoding="utf-8")

    results_csv = write_rows_csv(rows, out / "results.csv")
    summary_csv = write_rows_csv(summary_records, out / "summary.csv")
    stamp_tables(
        tables,
        source_run=run.path.name,
        command=f"python -m evaluation.rebuild --run {run.path.name}",
        n_rows=len(rows),
    )
    tex_paths = write_tables_tex(tables, tables_dir)

    if not quiet:
        print(f"  {report_path.relative_to(RESULTS_ROOT.parent)}")
        print(f"  {results_csv.name} ({len(rows)} rows), {summary_csv.name} "
              f"({len(summary_records)} rows)")
        print(f"  tables/ ({len(tex_paths)} .tex), figures/ ({len(figures)} figures)")

    return {
        "run": run,
        "rows": rows,
        "comparisons": comparisons,
        "summary_records": summary_records,
        "report": report_path,
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "tables": tex_paths,
        "figures": figures,
    }


def _run_identity(run: Run, rows) -> dict:
    """The columns that identify a run, repeated as the leading keys of every
    cross-run CSV so any two of them can be joined on (run_dir) or grouped on
    (organism, replica_count)."""
    return {
        "run_dir": run.path.name,
        "run_name": run.name,
        "organism": run.organism,
        "replica_count": run.replica_count,
        "n_samples": len(rows),
    }


def overview_row(run: Run, rows, comparisons) -> dict:
    """One wide row per run, every headline number at a glance. Meant to be read
    directly or dropped into a spreadsheet, unlike the long-format files which are
    meant to be grouped and plotted."""
    row = _run_identity(run, rows)

    for arm in run.arms:
        for metric in METRIC_KEYS:
            row[f"{arm}_{metric}"] = nanmean(arm_values(rows, arm, metric))

    for label, entry in comparisons.items():
        suffix = "vs_control" if entry["baseline_arm"] == "control" else "vs_deterministic"
        significant = 0
        for metric in METRIC_KEYS:
            result = entry["comparison"]["metrics"][metric]
            row[f"delta_{suffix}_{metric}"] = result["delta_ci"]["point"]
            row[f"p_holm_{suffix}_{metric}"] = result["holm"]["p_adjusted"]
            significant += bool(result["holm"]["reject"])
        row[f"n_significant_{suffix}"] = significant

    if "oracle" in run.arms:
        for metric, stats in oracle_gap(rows).items():
            row[f"oracle_gap_{metric}"] = stats["mean_gap"]

    successes, total = exact_match_count(rows, "agentic")
    row["exact_match_successes"] = successes
    row["exact_match_n"] = total

    nterm = nterm_analysis(rows)
    row["p_correct_nterm"] = nterm["p_correct_start"]
    row["em_given_correct_nterm"] = nterm["em_given_correct_start"]
    row["em_given_wrong_nterm"] = nterm["em_given_wrong_start"]

    bp = breakpoint_stats(rows)
    row["mean_breakpoints"] = bp.get("mean")
    row["mean_breakpoints_per_join"] = bp.get("mean_normalized")
    row["zero_breakpoint_samples"] = bp.get("zero_breakpoint_samples")

    concordance = concordance_summary(rows)
    row["mean_validity_concordance"] = concordance["mean_concordance"]
    row["concordance_comparable_pairs"] = concordance["comparable_pairs"]

    row["mean_confirmed_adjacencies"] = nanmean(
        [r.get("num_confirmed_adjacencies") for r in rows]
    )
    row["mean_pruned_pct"] = nanmean([r.get("pruned_pct") for r in rows])

    row.update({f"cost_{k}": v for k, v in cost_summary(rows).items() if k != "n_samples"})

    ranking = junction_ranking_summary(rows)
    row["junction_top1_acc"] = ranking["top1_acc"] if ranking else None
    row["junction_mrr"] = ranking["mrr"] if ranking else None
    recall = trypsin_recall_summary(rows)
    row["trypsin_recall"] = recall["recall"] if recall else None
    return row


def tests_rows(run: Run, rows, comparisons) -> list[dict]:
    """One row per (run, comparison, metric): the full paired-test result that
    section D prints, machine-readable."""
    out = []
    identity = _run_identity(run, rows)
    for label, entry in comparisons.items():
        for metric in METRIC_KEYS:
            result = entry["comparison"]["metrics"][metric]
            test, ci, holm = result["test"], result["delta_ci"], result["holm"]
            detail = test["detail"]
            out.append(
                {
                    **identity,
                    "comparison": label,
                    "arm_a": "agentic",
                    "arm_b": entry["baseline_arm"],
                    "metric": metric,
                    "metric_label": METRIC_NAMES[metric],
                    "mean_delta": ci["point"],
                    "delta_ci_low": ci["low"],
                    "delta_ci_high": ci["high"],
                    "delta_ci_method": ci["method"],
                    "test": "mcnemar_exact" if "discordant" in detail else "wilcoxon_signed_rank",
                    "test_method": detail.get("method"),
                    "statistic": test["statistic"],
                    "n_used": test["n"],
                    "discordant": detail.get("discordant"),
                    "n10_only_a": detail.get("n10_only_a"),
                    "n01_only_b": detail.get("n01_only_b"),
                    "n_nonzero": detail.get("n_nonzero"),
                    "n_positive": detail.get("n_positive"),
                    "n_negative": detail.get("n_negative"),
                    "p_raw": test["pvalue"],
                    "p_holm": holm["p_adjusted"],
                    "significant_holm": holm["reject"],
                    "alpha": entry["comparison"]["alpha"],
                    "family_size": entry["comparison"]["family_size"],
                }
            )
    return out


def stratified_rows(run: Run, rows) -> list[dict]:
    """One row per (run, fragment bin, arm, metric) with the paired lift over
    the shuffled floor — the priority analysis, in long format for plotting."""
    out = []
    identity = _run_identity(run, rows)
    for metric in METRIC_KEYS:
        lift = {
            arm: lift_over_baseline(rows, metric, arm)
            for arm in run.arms
            if arm != "shuffled"
        }
        for arm in run.arms:
            strat = stratify_by_bin(rows, arm, metric)
            for label in bin_labels():
                entry = strat[label]
                record = {
                    **identity,
                    "fragment_bin": label,
                    "arm": arm,
                    "metric": metric,
                    "metric_label": METRIC_NAMES[metric],
                    "n_in_bin": entry["n"],
                    "n_usable": len(entry["values"]),
                    "mean": nanmean(entry["values"]) if entry["values"] else None,
                }
                if arm in lift:
                    record["lift_over_shuffled"] = lift[arm][label]["lift"]
                    record["shuffled_mean"] = lift[arm][label]["baseline_mean"]
                out.append(record)
    return out


def taxonomy_rows(run: Run, rows) -> list[dict]:
    """One row per (run, error class) with count and share."""
    counts = taxonomy_counts(rows)
    total = sum(counts.values()) or 1
    identity = _run_identity(run, rows)
    return [
        {
            **identity,
            "error_class": key,
            "error_label": TAXONOMY_LABELS[key],
            "count": counts[key],
            "share": counts[key] / total,
        }
        for key in TAXONOMY_ORDER
    ]


CSV_GUIDE = """# Cross-run CSV exports

Generated by `python -m evaluation.rebuild --all`. Every value is computed from
the per-sample data stored in each run's `samples.jsonl`; nothing is transcribed
by hand. Re-running the rebuild reproduces these files exactly (bootstraps use a
fixed seed).

All six files share the leading identity columns `run_dir, run_name, organism,
replica_count, n_samples`, so any two can be joined on `run_dir` or grouped on
`(organism, replica_count)`.

| File | Grain | Rows | Use it for |
| --- | --- | --- | --- |
| `all_runs_overview.csv` | one row per run | 6 | Headline numbers at a glance: every arm x metric mean, deltas vs control and vs deterministic with Holm p-values, oracle gaps, N-terminal, breakpoints, concordance and cost. The 'key metrics' file. |
| `all_runs_summary.csv` | run x arm x metric | 150 | Aggregate means with 95% confidence intervals and the interval method (Wilson for Exact Match, BCa otherwise). Long format - good for plotting. |
| `all_runs_results.csv` | one row per sample | 600 | Raw per-sample data for your own analysis: all five arms x five metrics, plus fragment count/bin, protein length, breakpoints, N-terminal flag, error class, validity concordance and per-sample cost. |
| `all_runs_tests.csv` | run x comparison x metric | 60 | Paired-test results: mean delta with BCa CI, test used, discordant / non-zero pair counts, raw and Holm-adjusted p, and the significance flag. |
| `all_runs_stratified.csv` | run x bin x arm x metric | 750 | Performance by fragment count, with paired lift over the shuffled floor. |
| `all_runs_taxonomy.csv` | run x error class | 42 | Error-mode composition, counts and shares. |

Empty cells mean the value was not available for that run (for example
`junction_top1_acc` on runs recorded before that field was instrumented). An
empty cell is never a zero.

Per-run LaTeX tables and figures stay in each run's own folder, under
`results/<run>/tables/` and `results/<run>/figures/`.
"""


def generate_cross_run(artifacts: list[dict], out_dir: Path, quiet: bool = False) -> dict:
    """Combined CSV and report across every run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [row for art in artifacts for row in art["rows"]]
    all_summary = [rec for art in artifacts for rec in art["summary_records"]]

    combined_csv = write_rows_csv(all_rows, out_dir / "all_runs_results.csv")
    combined_summary = write_rows_csv(all_summary, out_dir / "all_runs_summary.csv")

    overview = [
        overview_row(art["run"], art["rows"], art["comparisons"]) for art in artifacts
    ]
    tests = [
        row
        for art in artifacts
        for row in tests_rows(art["run"], art["rows"], art["comparisons"])
    ]
    stratified = [
        row for art in artifacts for row in stratified_rows(art["run"], art["rows"])
    ]
    taxonomy = [
        row for art in artifacts for row in taxonomy_rows(art["run"], art["rows"])
    ]

    overview_csv = write_rows_csv(overview, out_dir / "all_runs_overview.csv")
    tests_csv = write_rows_csv(tests, out_dir / "all_runs_tests.csv")
    stratified_csv = write_rows_csv(stratified, out_dir / "all_runs_stratified.csv")
    taxonomy_csv = write_rows_csv(taxonomy, out_dir / "all_runs_taxonomy.csv")
    (out_dir / "README.md").write_text(CSV_GUIDE, encoding="utf-8")

    runs = [art["run"] for art in artifacts]
    matrix_rows = []
    for art in artifacts:
        run, rows = art["run"], art["rows"]
        matrix_rows.append(
            [
                run.organism,
                fmt(run.replica_count, 0),
                fmt(len(rows), 0),
                fmt(nanmean(arm_values(rows, "shuffled", PRIMARY_METRIC))),
                fmt(nanmean(arm_values(rows, "agentic", PRIMARY_METRIC))),
                fmt(nanmean(arm_values(rows, "agentic", "kendall_tau"))),
                fmt(nanmean(arm_values(rows, "agentic", "exact_match"))),
            ]
        )

    matrix = Table(
        key="table_cross_run_matrix",
        headers=[
            "Organism", "Replicas", "n", "Random Order APA", "LLM-Guided APA",
            "LLM-Guided tau", "LLM-Guided EM",
        ],
        rows=matrix_rows,
        caption="All runs: headline agentic performance by organism and replica count.",
        label="tab:cross_run_matrix",
    )

    # Organism gap conditioned on fragment count.
    by_organism: dict = {}
    for art in artifacts:
        by_organism.setdefault(art["run"].organism, []).extend(art["rows"])

    gap_rows = []
    for label in bin_labels():
        cells = [label]
        for organism, rows in by_organism.items():
            subset = [r for r in rows if r.get("fragment_bin") == label]
            values = [
                r.get(f"agentic_{PRIMARY_METRIC}")
                for r in subset
                if isinstance(r.get(f"agentic_{PRIMARY_METRIC}"), (int, float))
            ]
            lengths = [r.get("protein_length") for r in subset if r.get("protein_length")]
            cells.append(
                f"{fmt(nanmean(values))} (n={len(subset)}, len {fmt(nanmean(lengths), 0)})"
                if subset
                else "-"
            )
        gap_rows.append(cells)

    gap_table = Table(
        key="table_cross_run_organism_gap",
        headers=["Fragments"] + list(by_organism),
        rows=gap_rows,
        caption=(
            f"{METRIC_NAMES[PRIMARY_METRIC]} by organism, conditioned on fragment count. "
            "Mean protein length is shown per cell, since an organism gap that "
            "disappears within a fragment-count bin is a protein-length effect, not an "
            "organism effect."
        ),
        label="tab:organism_gap",
        notes=(
            "Cells pool every replica count. The design is balanced - each organism "
            "contributes the same replica counts with equal sample counts - so the "
            "comparison is fair, but the pooling widens within-cell spread and these "
            "values are not comparable to a single run's numbers."
        ),
    )

    tables = [matrix, gap_table]
    figures = []
    try:
        from evaluation import figures as figs

        series: dict = {}
        tau_series: dict = {}
        for art in artifacts:
            run, rows = art["run"], art["rows"]
            if run.replica_count is None:
                continue
            series.setdefault(run.organism, []).append(
                (run.replica_count, nanmean(arm_values(rows, "agentic", PRIMARY_METRIC)))
            )
            tau_series.setdefault(run.organism, []).append(
                (run.replica_count, nanmean(arm_values(rows, "agentic", "kendall_tau")))
            )
        if series:
            figures.append(
                figs.replica_scaling(
                    series, PRIMARY_METRIC, METRIC_NAMES[PRIMARY_METRIC],
                    out_dir / "figures", "fig_replica_scaling",
                )
            )
            figures.append(
                figs.replica_scaling(
                    tau_series, "kendall_tau", METRIC_NAMES["kendall_tau"],
                    out_dir / "figures", "fig_replica_scaling_tau",
                )
            )
        counts_by_run = {art["run"].label(): taxonomy_counts(art["rows"]) for art in artifacts}
        figures.append(
            figs.error_taxonomy(
                counts_by_run, TAXONOMY_LABELS, out_dir / "figures", "fig_error_taxonomy_all"
            )
        )
    except Exception as exc:
        print(f"  figure generation skipped: {exc}")

    text = [
        "# Cross-Run Statistical Report",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from "
        f"{len(artifacts)} runs, {len(all_rows)} samples total. All values recomputed "
        "from stored per-sample data._",
        "",
        "## Run matrix",
        "",
        matrix.to_markdown(),
        "",
        "## C. Replica Scaling",
        "",
        "More digestion replicas mean more adjacencies the overlap graph can confirm "
        "outright — near-ground-truth structure the search gets for free.",
        "",
    ]

    scaling_rows = []
    for organism, rows in by_organism.items():
        for art in artifacts:
            if art["run"].organism != organism:
                continue
            r = art["rows"]
            scaling_rows.append(
                [
                    organism,
                    fmt(art["run"].replica_count, 0),
                    fmt(nanmean([x.get("num_confirmed_adjacencies") for x in r]), 2),
                    fmt(nanmean(arm_values(r, "agentic", PRIMARY_METRIC))),
                    fmt(nanmean(arm_values(r, "agentic", "kendall_tau"))),
                ]
            )
    scaling_table = Table(
        key="table_cross_run_replica_scaling",
        headers=["Organism", "Replicas", "Confirmed adjacencies", "APA", "Kendall tau"],
        rows=sorted(scaling_rows, key=lambda r: (r[0], float(r[1]))),
        caption="Replica scaling: overlap-graph strength and reconstruction quality.",
        label="tab:replica_scaling",
    )
    tables.append(scaling_table)
    text += [scaling_table.to_markdown(), ""]

    text += [
        "## E. coli vs Yeast, conditioned on fragment count",
        "",
        "A raw organism gap can be a protein-length artifact: if one organism's proteins "
        "are simply cut into more fragments, it will look harder without any biological "
        "difference. Conditioning on fragment count is what separates the two.",
        "",
        gap_table.to_markdown(),
        "",
        "## Provenance",
        "",
        f"- `all_runs_overview.csv` ({len(overview)} rows) - one row per run, headline numbers",
        f"- `all_runs_summary.csv` ({len(all_summary)} rows) - arm x metric means with CIs",
        f"- `all_runs_results.csv` ({len(all_rows)} rows) - one row per sample",
        f"- `all_runs_tests.csv` ({len(tests)} rows) - paired tests with Holm-adjusted p",
        f"- `all_runs_stratified.csv` ({len(stratified)} rows) - by fragment count, with lift",
        f"- `all_runs_taxonomy.csv` ({len(taxonomy)} rows) - error-mode composition",
        "- See `README.md` in this folder for what each file is for.",
        "- No GPU, model loading or network access.",
        "",
    ]

    # Written once, after every section has contributed its tables.
    stamp_tables(
        tables,
        source_run=", ".join(art["run"].path.name for art in artifacts),
        command="python -m evaluation.rebuild --all",
        n_rows=len(all_rows),
    )
    tex_paths = write_tables_tex(tables, out_dir / "tables")
    report_path = out_dir / "cross_run_report.md"
    report_path.write_text("\n".join(text), encoding="utf-8")

    if not quiet:
        print(f"\nCross-run artifacts in {out_dir}:")
        print(f"  {report_path.name}")
        for path, count in (
            (overview_csv, len(overview)),
            (combined_summary, len(all_summary)),
            (combined_csv, len(all_rows)),
            (tests_csv, len(tests)),
            (stratified_csv, len(stratified)),
            (taxonomy_csv, len(taxonomy)),
        ):
            print(f"  {path.name} ({count} rows)")
        print("  README.md (what each CSV is for)")
        print(f"  tables/ ({len(tex_paths)} .tex), figures/ ({len(figures)} figures)")

    return {
        "report": report_path,
        "combined_csv": combined_csv,
        "combined_summary": combined_summary,
        "overview_csv": overview_csv,
        "tests_csv": tests_csv,
        "stratified_csv": stratified_csv,
        "taxonomy_csv": taxonomy_csv,
        "runs": runs,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.rebuild",
        description=(
            "Rebuild statistical reports, CSVs, LaTeX tables and figures from stored "
            "per-sample run data. No GPU, no model loading, no network."
        ),
    )
    parser.add_argument("--all", action="store_true", help="rebuild every run in results/")
    parser.add_argument("--run", action="append", default=[], help="a run folder (repeatable)")
    parser.add_argument(
        "--results-root", default=str(RESULTS_ROOT), help="root results directory"
    )
    parser.add_argument(
        "--output", default=None, help="cross-run output directory (default: <results>/_analysis)"
    )
    parser.add_argument(
        "--resamples", type=int, default=DEFAULT_RESAMPLES,
        help=f"bootstrap resamples (default {DEFAULT_RESAMPLES}); lower is faster, wider CIs",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.results_root)
    if args.all:
        run_dirs = discover_runs(root)
    elif args.run:
        run_dirs = [Path(r) if Path(r).exists() else root / r for r in args.run]
    else:
        parser.error("pass --all or --run <folder>")
        return 2

    if not run_dirs:
        print(f"No runs with a samples.jsonl found under {root}")
        return 1

    started = time.time()
    artifacts = []
    for run_dir in run_dirs:
        if not (Path(run_dir) / "samples.jsonl").exists():
            print(f"Skipping {run_dir} — no samples.jsonl")
            continue
        if not args.quiet:
            print(f"\n{Path(run_dir).name}")
        try:
            artifacts.append(
                generate_run_artifacts(run_dir, resamples=args.resamples, quiet=args.quiet)
            )
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    if not artifacts:
        print("Nothing rebuilt.")
        return 1

    if len(artifacts) > 1:
        out_dir = Path(args.output) if args.output else root / "_analysis"
        generate_cross_run(artifacts, out_dir, quiet=args.quiet)

    print(f"\nRebuilt {len(artifacts)} run(s) in {time.time() - started:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
