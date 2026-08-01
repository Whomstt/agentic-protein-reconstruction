"""How protein shape predicts reconstruction quality, per organism.

The stratification table shows quality falling as fragment count rises, but
fragment count and mean fragment length are confounded in the raw data: both are
functions of protein length and of where trypsin happened to cut. This module
separates them, correlating each against the three reported metrics so the
report can say which one difficulty actually tracks.

A composition layer, in the same spirit as thesis_tables.py: the correlation and
its p value come from evaluation/stats.py, the Holm correction from the same
place the paired tests use, the rendering from exports.py and figures.py. No GPU,
no model loading, no network - everything derives from samples.jsonl.

    python -m evaluation.fragment_correlation                    # both r100 runs
    python -m evaluation.fragment_correlation --run A --run B    # explicit runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.analysis import (
    RESULTS_ROOT,
    Run,
    bin_labels,
    discover_runs,
    fragment_bin,
    load_run,
)
from evaluation.exports import Table, fmt, fmt_p
from evaluation.stats import holm_bonferroni, spearman
from evaluation import figures

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLE_OUT = PROJECT_ROOT / "report" / "tables"
DEFAULT_FIGURE_OUT = PROJECT_ROOT / "report" / "images"

# PNG only, alongside the report's hand-made diagrams. The rest of the pipeline
# emits a vector PDF with a PNG twin for the markdown preview; here the paper
# includes the PNG directly, so a PDF would just be an unused second copy.
FIGURE_FORMATS = ("png",)

# The replica count the report's headline tables use, when runs are auto-selected.
DEFAULT_REPLICA_COUNT = 100

ALPHA = 0.05
PLACES = 3

# Matches thesis_tables.REPORTED_METRICS: the metrics the paper prints.
METRICS = (
    ("adjacent_pair_acc", "Adjacent Pair Accuracy"),
    ("exact_match", "Exact Match"),
    ("longest_correct_run", "Longest Correct Run"),
)

PREDICTORS = (
    ("num_fragments", "Fragments per protein"),
    ("mean_fragment_length", "Mean fragment length"),
)

SPECIES = {"ecoli": "E. coli", "yeast": "S. cerevisiae"}

# Equal-width bins for the figure, deliberately NOT analysis.FRAGMENT_BINS.
#
# Those bins widen as they climb (2-4, 5-9, ..., 50+), which suits a table where
# each row is read on its own but misleads on a plotted axis: evenly spaced ticks
# would represent unequal ranges, so the curve's slope would be an artifact of the
# binning. Equal width makes the spacing mean what it looks like it means.
#
# The width is set by where the proteins actually are, not by the range they span.
# Fragment counts reach 113 (E. coli) and 170 (yeast), but the distribution is
# heavily right-skewed: at width 20 the first bin swallows 64% of the E. coli
# sample and the early decay disappears into a single point. Width 10 to 80 keeps
# every bin populated on both organisms and leaves only six proteins of 200 off
# the axis - figure only, since every correlation, table and headline number in
# the report is computed on the full sample.
FIGURE_BIN_WIDTH = 10
FIGURE_BIN_MAX = 80

# The arm under test. Correlating the LLM-Guided arm asks what the shipped
# system's difficulty profile is; the floor arm's would only re-describe chance.
ARM_KEY = "recon_metrics"
ARM_LABEL = "LLM-Guided"


def _species(run: Run) -> str:
    key = (run.config.get("data") or {}).get("organism")
    return SPECIES.get(key, run.organism)


def predictors(sample: dict) -> dict:
    """Fragment count and mean fragment length for one protein.

    ``order`` is a permutation of the fragment set, so its length is the fragment
    count. ``target`` is the concatenation of those fragments, so the mean length
    is exact division rather than an estimate - which is why the fragment strings
    themselves do not need to be stored in samples.jsonl."""
    n_fragments = len(sample.get("order") or [])
    if not n_fragments:
        return {}
    return {
        "num_fragments": n_fragments,
        "mean_fragment_length": len(sample.get("target") or "") / n_fragments,
    }


def correlations(run: Run) -> dict:
    """{(predictor, metric): TestResult} for one run.

    Ordering metrics are NaN when a sample's fragments do not tile the target, and
    ``spearman`` drops those pairs, so n can differ per cell and is reported."""
    rows = []
    for sample in run.samples:
        shape = predictors(sample)
        if not shape:
            continue
        metrics = sample.get(ARM_KEY) or {}
        rows.append({**shape, **{key: metrics.get(key) for key, _ in METRICS}})

    out = {}
    for predictor, _ in PREDICTORS:
        for metric, _ in METRICS:
            out[(predictor, metric)] = spearman(
                [r[predictor] for r in rows],
                [r[metric] for r in rows],
                name=f"{predictor}~{metric}",
            )
    return out


def _holm(by_run: dict) -> dict:
    """Holm across every correlation reported, both organisms included.

    Wider than the per-comparison family the paired tests use, and deliberately so:
    these cells are one exploratory sweep over predictors, not a pre-registered
    hypothesis per organism, so the correction should cover the whole sweep."""
    raw = {
        f"{name}|{predictor}|{metric}": result.pvalue
        for name, results in by_run.items()
        for (predictor, metric), result in results.items()
    }
    return holm_bonferroni(raw, alpha=ALPHA)


def build_table(runs: list[Run], by_run: dict, adjusted: dict) -> Table:
    rows: list = []
    for predictor, predictor_label in PREDICTORS:
        for metric, metric_label in METRICS:
            cells = [predictor_label, metric_label]
            for run in runs:
                result = by_run[run.name][(predictor, metric)]
                verdict = adjusted.get(f"{run.name}|{predictor}|{metric}") or {}
                cells += [
                    fmt(result.statistic, PLACES, signed=True),
                    fmt_p(verdict.get("p_adjusted")),
                    str(result.n),
                ]
            rows.append(cells)

    headers = ["Predictor", "Metric"]
    for run in runs:
        headers += [
            rf"\textit{{{_species(run)}}} $\rho$",
            r"adj. $p$",
            "$n$",
        ]

    return Table(
        key="fragment_correlation_r100",
        headers=headers,
        rows=rows,
        column_spec="ll" + "rrr" * len(runs),
        environment="table*",
        placement="!t",
        raw_latex=True,
        caption=(
            "Spearman rank correlation between protein shape and reconstruction "
            f"quality for the {ARM_LABEL} arm, at "
            f"{runs[0].replica_count} digestion replicas. Negative $\\rho$ means "
            "quality falls as the predictor rises. Rank correlation is used because "
            "fragment count is right-skewed, Exact Match is binary and the ordering "
            "metrics saturate."
        ),
        label="tab:fragment_correlation",
        notes=(
            f"$p$ values are Holm-corrected across all {len(PREDICTORS) * len(METRICS) * len(runs)} "
            "correlations in the table. $n$ is the proteins with that metric defined; "
            "ordering metrics are undefined where the true fragment order could not be "
            "recovered."
        ),
    )


def figure_bins() -> list[tuple[int, int]]:
    """Equal-width (low, high] fragment-count intervals, inclusive of both ends as
    labelled: (0, 20] is printed 1-20."""
    return [
        (low, low + FIGURE_BIN_WIDTH)
        for low in range(0, FIGURE_BIN_MAX, FIGURE_BIN_WIDTH)
    ]


def build_figure(runs: list[Run], out_dir: Path) -> dict:
    """All three reported metrics against fragment count, one panel per organism."""
    bins = figure_bins()
    centres = [(low + high) / 2 for low, high in bins]
    panels = []
    excluded = 0
    for run in runs:
        binned: list = [[] for _ in bins]
        for sample in run.samples:
            shape = predictors(sample)
            if not shape:
                continue
            count = shape["num_fragments"]
            for index, (low, high) in enumerate(bins):
                if low < count <= high:
                    binned[index].append(sample.get(ARM_KEY) or {})
                    break
            else:
                excluded += 1

        series = []
        for metric, metric_label in METRICS:
            values = []
            for members in binned:
                # NaN ordering metrics are dropped, so a bin's mean can rest on
                # fewer proteins than its n; an empty bin becomes a gap.
                scores = [
                    m[metric] for m in members
                    if m.get(metric) is not None and m[metric] == m[metric]
                ]
                values.append(sum(scores) / len(scores) if scores else None)
            series.append((metric_label, values))

        panels.append((_species(run), centres, series))

    if excluded:
        print(
            f"note: {excluded} protein(s) above {FIGURE_BIN_MAX} fragments are off the "
            "figure's axis; correlations and tables still use every sample."
        )
    return figures.metrics_by_fragment_bin(
        panels, out_dir, "fragment_correlation", FIGURE_FORMATS
    )


def resolve_runs(explicit: list[str], results_root: Path, replica_count: int) -> list[Run]:
    """Explicit run folders if given, else one run per organism at
    ``replica_count`` - the configuration the report's headline tables use."""
    if explicit:
        runs = []
        for value in explicit:
            path = Path(value) if Path(value).exists() else results_root / value
            runs.append(load_run(path))
        return runs

    by_organism: dict = {}
    for path in discover_runs(results_root):
        run = load_run(path)
        if run.replica_count == replica_count:
            by_organism[run.organism] = run  # later folder wins; names sort oldest first
    return list(by_organism.values())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.fragment_correlation",
        description=(
            "Correlate fragment count and mean fragment length against the reported "
            "metrics, per organism. Writes a booktabs table and a figure."
        ),
    )
    parser.add_argument(
        "--run", action="append", default=[],
        help="run folder under results/ (repeatable); default is one run per organism",
    )
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument(
        "--replica-count", type=int, default=DEFAULT_REPLICA_COUNT,
        help=f"replica count to auto-select runs at (default {DEFAULT_REPLICA_COUNT})",
    )
    parser.add_argument("--table-out", default=str(DEFAULT_TABLE_OUT))
    parser.add_argument("--figure-out", default=str(DEFAULT_FIGURE_OUT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    runs = resolve_runs(args.run, Path(args.results_root), args.replica_count)
    if len(runs) < 1:
        print("No runs found — pass --run explicitly.")
        return 1

    by_run = {run.name: correlations(run) for run in runs}
    adjusted = _holm(by_run)

    table = build_table(runs, by_run, adjusted)
    table_path = Path(args.table_out) / f"{table.key}.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    # Camera-ready, matching thesis_tables: these are \input{} into the paper.
    table_path.write_text(table.to_latex(comments=False), encoding="utf-8")

    written = [table_path]
    if figures.available():
        written += list(build_figure(runs, Path(args.figure_out)).values())
    else:
        print("matplotlib unavailable — figure skipped, table still written.")

    if not args.quiet:
        for run in runs:
            print(f"\n{_species(run)} ({run.name}, n={run.n})")
            for predictor, predictor_label in PREDICTORS:
                for metric, metric_label in METRICS:
                    result = by_run[run.name][(predictor, metric)]
                    verdict = adjusted.get(f"{run.name}|{predictor}|{metric}") or {}
                    star = "*" if verdict.get("reject") else " "
                    print(
                        f"  {predictor_label:22s} vs {metric_label:24s} "
                        f"rho={result.statistic:+.3f}  adj p={verdict.get('p_adjusted', float('nan')):.4f}{star}"
                        f"  n={result.n}"
                    )
        print()
        for path in written:
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
