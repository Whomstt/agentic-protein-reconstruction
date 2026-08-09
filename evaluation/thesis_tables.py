"""Publication-format LaTeX tables for the report's Results section.

    python -m evaluation.thesis_tables --all
    python -m evaluation.thesis_tables --run 130726_224804_agentic

Each configuration writes its own set, named and labelled for the setup it came
from (``main_results_ecoli_r100.tex`` -> ``tab:main_results_ecoli_r100``), so all
six sit in ``report/tables`` together. Cells are point estimates: the confidence
intervals and their bounds live in ``analysis_report.md``.

Every number is computed from that run's ``samples.jsonl``; nothing is read out
of a generated report or CSV, and nothing is transcribed by hand. Files are
written camera-ready (no provenance comments) into ``report/tables``.

This is a composition layer, not a second statistics implementation: the
aggregations come from ``evaluation/analysis.py``, the intervals and paired tests
from ``evaluation/rebuild.py``, the rendering from ``evaluation/exports.py``. So
these tables and ``analysis_report.md`` cannot disagree. The exception is the
agent-behaviour table, computed nowhere else, which reads the per-iteration
``lever_values`` / ``changed_levers`` / ``validity_score`` records.

No GPU, no model loading, no network.
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
    samples_path,
    stratify_by_bin,
    taxonomy_counts,
)
from evaluation.exports import (
    MIDRULE,
    Raw,
    Table,
    fmt,
    fmt_p,
    latex_escape,
    stamp_tables,
    write_tables_tex,
)
from evaluation.metrics import METRIC_NAMES, nanmean

# Imported, not reimplemented, so these tables and analysis_report.md agree.
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

# Paper precision: 3 places is already past the resolution the CIs support.
PLACES = 3

LEVERS = ("junction_window", "search_mode", "beam_width", "edge_mode", "confirmed_bonus")

# Column headings for the headline table, where the full arm labels do not fit.
SHORT_ARM_LABELS = {
    "shuffled": "Random Order",
    "deterministic": "Fixed Settings",
    "control": "Random Search",
    "agentic": "LLM-Guided",
    "oracle": "Best Candidate",
}

# The metrics the report's two headline tables PRINT, in reading order: the
# primary ordering metric first, then the two that qualify it.
#
# Every metric in METRIC_KEYS is still computed and still corrected against, so
# this is a display filter and nothing else. Sequence Similarity is bought
# largely by fragment composition, which every arm shares; Kendall Tau moves
# with Adjacent Pair Accuracy and adds no independent evidence at this width.
# Both remain in analysis_report.md and in the selection-ceiling table.
#
# Edit Similarity is printed alongside them as the residue-level view: how much
# of the protein sits in the right place, on a standard edit distance rather than
# difflib's matching-block heuristic. Its Random Order cell is empty by
# construction — see analysis._backfill_edit_similarity.
REPORTED_METRICS = (
    "adjacent_pair_acc",
    "exact_match",
    "longest_correct_run",
    "edit_similarity",
)

# The Holm family is the full set of metrics tested, NOT the subset printed:
# narrowing the printed rows must never relax the correction the shown p values
# already carry.
HOLM_FAMILY_SIZE = len(METRIC_KEYS)

# Species names for the configuration table; the configs carry short keys.
SPECIES = {"ecoli": "E. coli", "yeast": "S. cerevisiae"}


def _point(interval) -> str:
    """The mean alone. Confidence intervals are computed and reported in
    ``analysis_report.md``; the report's tables print point estimates only."""
    if interval is None:
        return "n/a"
    data = interval if isinstance(interval, dict) else interval.as_dict()
    return fmt(data.get("point"), PLACES)


def run_suffix(run: Run) -> str:
    """The per-setup filename/label suffix, e.g. ``_ecoli_r100``.

    Every configuration in the grid writes its own tables, so a name has to say
    which organism and replica count it came from.
    """
    organism = (run.config.get("data") or {}).get("organism") or "run"
    return f"_{organism}_r{run.replica_count}"


def _setup(run: Run) -> str:
    """'\\textit{E. coli}, 100 replicas' - the caption's setup identifier."""
    return rf"{_species_it(run)}, {run.replica_count} digestion replicas"


def _species_it(run: Run) -> str:
    """The italicised binomial, safe inside IEEEtran's small-caps captions.

    ``\\textit`` on its own asks for a small-caps italic Times shape that does
    not exist, so LaTeX warns and substitutes; ``\\textnormal`` resets the
    family first and reaches the same italic without the warning.
    """
    return rf"\textnormal{{\textit{{{_species(run)}}}}}"


def _species(run: Run) -> str:
    """The binomial for a caption that italicises it.

    ``run.organism`` is a display name meant for prose ("Yeast"), which must not
    go inside \\textit{}; the binomial is what belongs there.
    """
    key = (run.config.get("data") or {}).get("organism")
    return SPECIES.get(key, run.organism)

# A table whose caption, notes or headers need maths or font commands sets
# raw_latex=True and writes them as LaTeX; data cells are always escaped.


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
    """I - the headline table: every arm, every metric."""
    arms = run.arms
    table_rows = []
    for metric in REPORTED_METRICS:
        cells = [METRIC_NAMES[metric]]
        for arm in arms:
            cells.append(_point(metric_interval(rows, arm, metric, resamples)))
        table_rows.append(cells)

    return Table(
        key="main_results",
        headers=["Metric"] + [SHORT_ARM_LABELS[a] for a in arms],
        rows=table_rows,
        caption=(
            rf"Reconstruction quality on {_species_it(run)} at "
            rf"{run.replica_count} digestion replicas ($n={len(rows)}$ proteins). "
            r"Mean per protein."
        ),
        label="tab:main_results",
        # The full metric family is disclosed by the paired-tests table's note,
        # which is where the Holm correction it matters for is actually reported.
        notes=(
            rf"{METRIC_NAMES[PRIMARY_METRIC]} is the primary metric. Edit "
            r"Similarity was added post these runs and is n/a for Random Order, "
            r"whose sequences were not stored."
        ),
        environment="table*",
        placement="!tb",
        raw_latex=True,
    )


def table_main_results_all(runs: list[Run], resamples: int = DEFAULT_RESAMPLES) -> Table:
    """The headline table for every configuration at once, one block per run.

    Same cells as :func:`table_main_results`, computed the same way from the same
    ``metric_interval``; the only difference is that all six setups share one
    float, with a ``\\midrule`` between blocks the way the organism-gap table
    separates its replica blocks. Organism and replica count are printed on each
    block's first row only, so the eye reads down the leading columns as group
    labels rather than as repeated data.

    Runs are ordered organism-major, then by descending replica count, so the
    reading order matches the prose: the richest condition first, then the two
    increasingly constrained ones, then the same three for the second organism.
    """
    ordered = sorted(runs, key=lambda r: (_species(r), -r.replica_count))
    arms: list[str] = []
    for run in ordered:
        for arm in run.arms:
            if arm not in arms:
                arms.append(arm)

    table_rows: list = []
    for run in ordered:
        rows = sample_rows(run)
        if table_rows:
            table_rows.append(MIDRULE)  # one rule per configuration block
        for position, metric in enumerate(REPORTED_METRICS):
            cells = [
                Raw(rf"\textit{{{_species(run)}}}", _species(run)) if position == 0 else "",
                str(run.replica_count) if position == 0 else "",
                METRIC_NAMES[metric],
            ]
            for arm in arms:
                cells.append(
                    _point(metric_interval(rows, arm, metric, resamples))
                    if arm in run.arms else "n/a"
                )
            table_rows.append(cells)

    return Table(
        key="main_results_all",
        headers=["Organism", "Replicas", "Metric"] + [SHORT_ARM_LABELS[a] for a in arms],
        rows=table_rows,
        column_spec="lrl" + "r" * len(arms),
        caption=(
            "Reconstruction quality across every configuration "
            f"($n={len(sample_rows(ordered[0]))}$ proteins each). Mean per protein."
        ),
        label="tab:main_results_all",
        # The full metric family is disclosed by the paired-tests table's note,
        # which is where the Holm correction it matters for is actually reported.
        notes=(
            rf"{METRIC_NAMES[PRIMARY_METRIC]} is the primary metric. Edit "
            r"Similarity was added post these runs and is n/a for Random Order, "
            r"whose sequences were not stored. Replicas is the number of "
            r"digestion replicas the overlap graph was built from."
        ),
        environment="table*",
        placement="!tb",
        raw_latex=True,
    )


def table_paired_tests(run: Run, rows, comparisons) -> Table:
    """II - the significance table. Both paired comparisons in one float.

    Every metric is tested and Holm-corrected across the family of five; the
    table prints the three reported ones. The p values shown are the ones the
    five-metric correction produced - narrowing the printed rows does not
    recompute Holm over three, which would silently make every p smaller.
    """
    table_rows = []
    for label, entry in comparisons.items():
        if table_rows:
            table_rows.append(MIDRULE)
        comparison = entry["comparison"]
        for i, metric in enumerate(REPORTED_METRICS):
            result = comparison["metrics"][metric]
            test, ci, holm = result["test"], result["delta_ci"], result["holm"]
            detail = test["detail"]
            if "discordant" in detail:
                pairs = (
                    f"{detail['discordant']} "
                    f"({detail['n10_only_a']}/{detail['n01_only_b']})"
                )
            else:
                pairs = (
                    f"{detail['n_nonzero']} "
                    f"({detail['n_positive']}/{detail['n_negative']})"
                )
            table_rows.append(
                [
                    label if i == 0 else "",
                    METRIC_NAMES[metric],
                    fmt(ci["point"], PLACES, signed=True),
                    pairs,
                    fmt_p(holm["p_adjusted"]),
                ]
            )

    return Table(
        key="paired_tests",
        headers=[
            r"Comparison", r"Metric", r"Mean $\Delta$",
            r"Proteins (+/-)", r"Adjusted $p$",
        ],
        rows=table_rows,
        column_spec="llrrr",
        caption=(
            rf"Paired per-sample comparisons on {_species_it(run)} at "
            rf"{run.replica_count} replicas ($n={len(rows)}$). Mean difference, "
            r"with $p$ values Holm-corrected within each comparison."
        ),
        label="tab:paired_tests",
        notes=(
            rf"Proteins: discordant pairs (LLM-Guided-only/baseline-only) for Exact "
            rf"Match, non-zero differences (positive/negative) otherwise. The Holm "
            rf"adjustment covers the {HOLM_FAMILY_SIZE} metrics computed, not only "
            rf"the {len(REPORTED_METRICS)} shown."
        ),
        environment="table*",
        placement="!tb",
        raw_latex=True,
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
        key="stratification",
        headers=["Fragments", "n"] + [_arm_label(run, a) for a in arms] + ["Lift"],
        rows=table_rows,
        caption=(
            f"{METRIC_NAMES[PRIMARY_METRIC]} by fragment count. Difficulty scales with "
            "how many pieces a protein was digested into: the number of possible "
            "orderings grows factorially while the evidence available at each junction "
            "does not. Lift is the mean paired per-sample difference between the "
            "LLM-Guided arm and the Random Order floor within the bin, not a difference "
            rf"of two independent means. Setup: {_setup(run)}."
        ),
        label="tab:stratification",
        notes=(
            "n counts proteins in the bin; a bin's mean is taken over those whose "
            "ordering metrics are defined (true fragment order recovered), so it can "
            "rest on fewer than n."
        ),
        environment="table*",
        placement="!tb",
        raw_latex=True,
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
        key="selection_ceiling",
        headers=["Metric", "LLM-Guided", "Best Candidate", "Gap", "Samples with a gap"],
        rows=table_rows,
        caption=(
            "Selection ceiling. The Best Candidate column takes, per metric, the best "
            "candidate the agent had already generated, using ground truth to choose. "
            "The gap is "
            "therefore quality the run reached and then discarded - recoverable by a "
            "better selection signal alone, with no additional search. "
            rf"Setup: {_setup(run)}."
        ),
        label="tab:selection_ceiling",
        raw_latex=True,
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
        key="validity_concordance",
        headers=["Measurement", "Value"],
        rows=table_rows,
        caption=(
            "Trust in the selection signal. Within each sample, concordance is the "
            "fraction of candidate pairs whose validity ordering (lower is better) "
            f"agrees with their true {METRIC_NAMES[PRIMARY_METRIC]} ordering, across the "
            "iterations that sample tried. 0.50 is a coin flip. Since the run keeps "
            "whichever candidate scores best on this signal, its concordance bounds what "
            rf"the search can deliver. Setup: {_setup(run)}."
        ),
        raw_latex=True,
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
        key="agent_behaviour",
        headers=["Measurement", "Value"],
        rows=table_rows,
        caption=(
            "Agent behaviour across the iteration budget, computed from the per-iteration "
            "records. Iteration 1 runs the fixed default levers with no LLM call, so "
            "lever-change rates are taken over the LLM-driven iterations only. 'Kept "
            "candidate came from iteration 1' is the share of proteins on which no later "
            rf"attempt displaced that deterministic first pass. Setup: {_setup(run)}."
        ),
        label="tab:agent_behaviour",
        raw_latex=True,
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
        key="error_taxonomy",
        headers=["Failure mode", "Proteins", "Share"],
        rows=table_rows,
        caption=(
            "Error taxonomy of the LLM-Guided arm's reconstructions. Each protein falls in "
            "exactly one class, checked most-specific first, and classified from the "
            "stored metric values (which were computed with the correct fragment-string "
            "semantics rather than recounted from fragment indices). "
            rf"Setup: {_setup(run)}."
        ),
        label="tab:error_taxonomy",
        raw_latex=True,
        notes=(
            "The cut points separating these classes are disclosed, untuned round "
            "numbers; they change only how a failure is labelled, never any headline "
            "metric."
        ),
    )


def table_cost(run: Run, rows) -> Table:
    """VIII - what the agentic arm costs against the control arm.

    The control arm is a non-LLM lever policy, so its call and token counts are
    zero by construction and the time ratio is the price of the reasoning.
    """
    cost = cost_summary(rows)
    agentic_seconds = cost["agentic_seconds_per_sample"]
    control_seconds = cost["control_seconds_per_sample"]
    ratio = agentic_seconds / control_seconds if control_seconds else float("nan")

    table_rows = [
        ["LLM calls", fmt(cost["llm_calls_per_sample"], 2), "0"],
        ["LLM tokens", fmt(cost["llm_tokens_per_sample"], 1), "0"],
        ["Wall clock (s)", fmt(agentic_seconds, 1), fmt(control_seconds, 1)],
    ]
    return Table(
        key="cost",
        headers=["Measurement", "LLM-Guided", "Random Search"],
        rows=table_rows,
        caption=(
            rf"Cost per protein on {_species_it(run)} at {run.replica_count} "
            rf"digestion replicas ($n={run.n}$ proteins). The Random Search arm runs "
            "the same budget and pipeline with lever values from a non-LLM policy."
        ),
        label="tab:cost",
        notes=(
            rf"Time ratio {fmt(ratio, 2)}$\times$. Wall clock covers the full "
            r"per-protein pipeline, including the PLM scoring both arms share."
        ),
        # Three narrow columns: this one fits a single IEEE column, so it can sit
        # beside the paragraph that reads the time ratio off it.
        column_spec="@{}lrr@{}",
        environment="table",
        placement="!tb",
        body_size=r"\small",
        col_sep="4pt",
        raw_latex=True,
    )


def config_table_tex(runs: list[Run]) -> str:
    """The experimental-configuration table, derived from every run's config
    snapshot so it cannot drift from what was actually executed.

    Written from a literal IEEE-style template rather than through ``Table``:
    it is the one table in the report that is not booktabs.
    """

    def distinct(getter, key=None):
        seen = []
        for run in runs:
            value = getter(run)
            if value is not None and value not in seen:
                seen.append(value)
        return ", ".join(str(v) for v in sorted(seen, key=key or str))

    def config(run, section, field, default=None):
        return (run.config.get(section) or {}).get(field, default)

    organisms = distinct(
        lambda r: SPECIES.get(config(r, "data", "organism"), config(r, "data", "organism"))
    )
    rows = [
        ("Dataset", "UniProt Reviewed (Swiss-Prot)"),
        ("Organisms", ", ".join(rf"\textit{{{o}}}" for o in organisms.split(", "))),
        ("Digestion replica counts", distinct(lambda r: config(r, "data", "replica_count"), key=int)),
        ("Missed cleavage ratio", distinct(lambda r: config(r, "data", "missed_cleavage_ratio"))),
        ("Proteins per configuration", distinct(lambda r: len(r.samples), key=int)),
        ("Protein language model", rf"\texttt{{{latex_escape(distinct(lambda r: _model_short(config(r, 'mlm_model', 'name'))))}}}"),
        ("LLM", rf"\texttt{{{latex_escape(distinct(lambda r: config(r, 'llm_model', 'name')))}}}"),
        ("Iterations", distinct(lambda r: config(r, "search", "max_iterations"), key=int)),
        ("Random seed", distinct(lambda r: config(r, "misc", "seed"), key=int)),
    ]
    body = "\n".join(rf"{name} & {value} \\" for name, value in rows)
    # Two columns of short values: a single-column float, so it sits next to the
    # experimental-setup paragraph rather than taking a page top of its own.
    return (
        "\\begin{table}[!tb]\n"
        "\\caption{Experimental Configuration}\n"
        "\\label{tab:config}\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}ll@{}}\n"
        "\\hline\n"
        "\\textbf{Parameter} & \\textbf{Value} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\par\\smallskip\\footnotesize{Shared by every run reported; only the five "
        "search levers vary.}\n"
        "\\end{table}\n"
    )


def _model_short(name) -> str:
    """'facebook/esm2_t6_8M_UR50D' -> 'esm2_t6_8M'."""
    if not name:
        return "n/a"
    return str(name).split("/")[-1].removesuffix("_UR50D")


# The registry. Adding or removing a thesis table is one entry here.
BUILDERS = (
    ("main_results", lambda run, rows, ctx: table_main_results(run, rows, ctx["resamples"])),
    ("paired_tests", lambda run, rows, ctx: table_paired_tests(run, rows, ctx["comparisons"])),
    ("stratification", lambda run, rows, ctx: table_stratification(run, rows)),
    ("selection_ceiling", lambda run, rows, ctx: table_selection_ceiling(run, rows)),
    ("validity_concordance", lambda run, rows, ctx: table_validity_concordance(run, rows)),
    ("agent_behaviour", lambda run, rows, ctx: table_agent_behaviour(run, rows)),
    ("error_taxonomy", lambda run, rows, ctx: table_error_taxonomy(run, rows)),
    ("cost", lambda run, rows, ctx: table_cost(run, rows)),
)

# Tables that need an arm the run may not have produced.
REQUIRES_ARM = {
    "selection_ceiling": "oracle",
}


def _sibling_runs(run: Run, results_root: Path) -> list[Run]:
    """Every run under ``results_root``, for the configuration table: it reports
    the whole experimental grid, not just the run the result tables come from."""
    runs = []
    for path in sorted(Path(results_root).iterdir()):
        if samples_path(path) is not None:
            runs.append(run if path.name == run.path.name else load_run(path))
    return runs or [run]


def build_tables(run_dir, out_dir: Path, resamples: int = DEFAULT_RESAMPLES,
                 results_root: Path = RESULTS_ROOT, quiet: bool = False,
                 only: tuple[str, ...] | None = None, suffix: str = "auto") -> list[Path]:
    """Compute and write thesis tables for one run.

    ``only`` restricts the build to those table keys; ``suffix`` is appended to
    each table's filename and LaTeX label so every configuration's tables can sit
    in the same directory and be \\input into the same document without colliding
    on either. The default ``"auto"`` derives it from the run itself
    (``_ecoli_r100``); pass ``""`` for unsuffixed names.
    """
    run = load_run(run_dir)
    if suffix == "auto":
        suffix = run_suffix(run)
    rows = sample_rows(run)
    comparisons = paired_comparisons(run, rows)
    ctx = {"resamples": resamples, "comparisons": comparisons}

    if only:
        unknown = set(only) - {key for key, _ in BUILDERS}
        if unknown:
            raise SystemExit(f"unknown table key(s): {', '.join(sorted(unknown))}")

    tables: list[Table] = []
    skipped: list[str] = []
    for key, builder in BUILDERS:
        if only and key not in only:
            continue
        needed = REQUIRES_ARM.get(key)
        if needed and needed not in run.arms:
            skipped.append(f"{key} (run has no {needed} arm)")
            continue
        if key == "paired_tests" and not comparisons:
            skipped.append(f"{key} (run has no paired baseline arm)")
            continue
        table = builder(run, rows, ctx)
        if suffix:
            table.key += suffix
            if table.label:
                table.label += suffix
        tables.append(table)

    stamp_tables(
        tables,
        source_run=run.path.name,
        command=_command(run.path.name),
        n_rows=len(rows),
        source_file=f"results/{run.path.name}/samples.jsonl",
    )
    # Camera-ready: no provenance comments in the files the report inputs. The
    # index always merges, so building one configuration's tables never drops
    # another configuration's from it.
    partial = bool(only)
    paths = write_tables_tex(
        tables, out_dir, comments=False,
        extra_inputs=[] if partial else ["config_table"],
        merge_index=True,
    )

    if not partial:
        config_path = out_dir / "config_table.tex"
        config_path.write_text(config_table_tex(_sibling_runs(run, results_root)), encoding="utf-8")
        paths.append(config_path)

    if not quiet:
        print(f"{run.path.name} -> {out_dir}")
        for path in paths:
            print(f"  {path.name}")
        for note in skipped:
            print(f"  skipped: {note}")
    return paths


def build_cross_run_tables(run_dirs, out_dir: Path, resamples: int = DEFAULT_RESAMPLES,
                           quiet: bool = False) -> list[Path]:
    """Write the tables that span every configuration rather than one run.

    Unsuffixed by construction: there is exactly one of each, so there is nothing
    for a suffix to disambiguate.
    """
    runs = [load_run(path) for path in run_dirs]
    tables = [table_main_results_all(runs, resamples)]
    stamp_tables(
        tables,
        source_run=", ".join(run.path.name for run in runs),
        command="python -m evaluation.thesis_tables --all",
        n_rows=sum(len(run.samples) for run in runs),
        source_file="samples.jsonl in each run folder",
    )
    paths = write_tables_tex(tables, out_dir, comments=False, merge_index=True)
    if not quiet:
        print(f"cross-run -> {out_dir}")
        for path in paths:
            print(f"  {path.name}")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.thesis_tables",
        description=(
            "Generate the report's Results tables as booktabs LaTeX, computed from a "
            "run's samples.jsonl. No GPU, no model loading, no network."
        ),
    )
    parser.add_argument("--run", help="run folder under results/ (or a path)")
    parser.add_argument(
        "--all", action="store_true",
        help="build tables for every run under results/, one suffixed set per configuration",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument(
        "--resamples", type=int, default=DEFAULT_RESAMPLES,
        help=f"bootstrap resamples (default {DEFAULT_RESAMPLES})",
    )
    parser.add_argument(
        "--only", default="",
        help="comma-separated table keys to build (default: all)",
    )
    parser.add_argument(
        "--suffix", default="auto",
        help=(
            "appended to each table's filename and LaTeX label so configurations "
            "coexist; 'auto' (default) derives it from the run, e.g. _ecoli_r100 -> "
            "tab:main_results_ecoli_r100. Pass '' for unsuffixed names."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.results_root)
    if args.all:
        run_dirs = [p for p in sorted(root.iterdir()) if samples_path(p) is not None]
        if not run_dirs:
            print(f"No runs with samples.jsonl under {root}")
            return 1
    else:
        if not args.run:
            print("Pass --run <folder> or --all")
            return 1
        run_dir = Path(args.run) if Path(args.run).exists() else root / args.run
        if samples_path(run_dir) is None:
            print(f"No samples.jsonl under {run_dir}")
            return 1
        run_dirs = [run_dir]

    for run_dir in run_dirs:
        build_tables(
            run_dir,
            Path(args.out),
            resamples=args.resamples,
            results_root=root,
            quiet=args.quiet,
            only=tuple(k.strip() for k in args.only.split(",") if k.strip()) or None,
            suffix=args.suffix,
        )

    # Cross-run: one headline table covering every configuration, so the report
    # can show the whole grid in a single float. Needs every run, so it is built
    # here rather than in the per-run BUILDERS registry.
    if args.all and not args.only:
        build_cross_run_tables(run_dirs, Path(args.out), resamples=args.resamples, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
