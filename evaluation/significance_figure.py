"""Generate organism-specific r100 significance tables.

The figure is derived from stored per-sample results only. No model, network,
or manually transcribed statistic is involved.

    python -m evaluation.significance_figure --results-root final_results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.analysis import (
    RESULTS_ROOT,
    arm_values,
    discover_runs,
    load_run,
    sample_rows,
)
from evaluation.exports import Table, fmt, fmt_p, write_tables_tex
from evaluation.rebuild import compare_arms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLE_OUT = PROJECT_ROOT / "report" / "tables"
METRICS = (
    "adjacent_pair_acc",
    "exact_match",
    "longest_correct_run",
    "edit_similarity",
)
METRIC_LABELS = {
    "adjacent_pair_acc": "APA",
    "exact_match": "EM",
    "longest_correct_run": "LCR",
    "edit_similarity": "ES",
}
SPECIES_LATEX = {
    "ecoli": r"{\normalfont\itshape E. coli}",
    "yeast": r"{\normalfont\itshape S. cerevisiae}",
}


def _organism_key(run) -> str:
    return (run.config.get("data") or {}).get("organism") or run.organism


def resolve_runs(results_root: Path):
    runs = [load_run(path) for path in discover_runs(results_root)]
    return sorted(
        (run for run in runs if run.has_control and run.replica_count == 100),
        key=lambda run: (_organism_key(run), run.replica_count),
    )


def significance_records(run) -> list[dict]:
    rows = sample_rows(run)
    arm_a = {metric: arm_values(rows, "agentic", metric) for metric in METRICS}
    arm_b = {metric: arm_values(rows, "control", metric) for metric in METRICS}
    comparison = compare_arms(arm_a, arm_b, metrics=METRICS)
    records = []
    for metric in METRICS:
        result = comparison["metrics"][metric]
        records.append(
            {
                "metric": METRIC_LABELS[metric],
                "delta": result["delta_ci"]["point"],
                "p": result["test"]["pvalue"],
                "p_adjusted": result["holm"]["p_adjusted"],
                "significant": result["holm"]["reject"],
                "n": len(rows),
            }
        )
    return records


def build_table(run, records: list[dict], name: str) -> Table:
    return Table(
        key=name,
        headers=[
            "Metric",
            "Mean $\\Delta$",
            "$p$",
            "Holm-adjusted $p$",
            "Significant",
        ],
        rows=[
            [
                record["metric"],
                fmt(record["delta"], 3, signed=True),
                fmt_p(record["p"]),
                fmt_p(record["p_adjusted"]),
                "Yes" if record["significant"] else "No",
            ]
            for record in records
        ],
        caption=(
            f"Paired comparison of LLM-Guided and Random Search on "
            f"{SPECIES_LATEX[_organism_key(run)]} at 100 replicas ($n=100$)."
        ),
        label=f"tab:llm_random_significance_{_organism_key(run)}_r100",
        notes=(
            "P-values use paired Wilcoxon signed-rank tests, except Exact Match "
            "which uses exact McNemar's test."
        ),
        column_spec="lrrrr",
        environment="table",
        placement="!tb",
        body_size=r"\scriptsize",
        col_sep="4pt",
        raw_latex=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.significance_figure",
        description="Generate the paired APA significance table from stored results.",
    )
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--table-out", default=str(DEFAULT_TABLE_OUT))
    args = parser.parse_args(argv)

    runs = resolve_runs(Path(args.results_root))
    if len(runs) < 2:
        print("Both E. coli and yeast r100 runs with control arms are required.")
        return 1
    tables = []
    all_records = []
    for run in runs:
        records = significance_records(run)
        all_records.append((run, records))
        name = f"llm_random_significance_{_organism_key(run)}_r100"
        tables.append(build_table(run, records, name))
    written = write_tables_tex(tables, Path(args.table_out), comments=False)
    for run, records in all_records:
        print(f"{run.organism} r100")
        for record in records:
            print(
                f"  {record['metric']}: delta={record['delta']:+.4f} "
                f"p={record['p']:.6f} Holm p={record['p_adjusted']:.6f} "
                f"significant={record['significant']} "
                f"n={record['n']}"
            )
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
