"""Random Search vs LLM-Guided: the two matched-budget arms, side by side.

The control arm runs the same pipeline, the same iteration budget and the same
best-validity selection as the agent, with levers drawn by a non-LLM policy. It
is therefore the only comparison that isolates the LLM's reasoning, and these
figures plot exactly that pair and nothing else.

One figure per organism, at 100 digestion replicas — the configuration the
report's headline tables use — with every reported metric on the axis. Two
figures rather than one grouped chart because the two organisms are read
separately in the text and eight bars per panel is already the readable limit.

Cells are means over the run's samples, computed with ``nanmean`` so proteins
whose true order could not be recovered drop out rather than counting as zero.
No GPU, no model loading, no network: everything derives from samples.jsonl.

    python -m evaluation.search_comparison --results-root final_results
    python -m evaluation.search_comparison --replica-count 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.analysis import ARMS, RESULTS_ROOT, Run, discover_runs, load_run
from evaluation.metrics import METRIC_NAMES, nanmean
from evaluation import figures

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURE_OUT = PROJECT_ROOT / "report" / "images"

# PNG only: the paper includes the PNG directly, as with the other report-side
# figures, so a vector twin would just be an unused second copy.
FIGURE_FORMATS = ("png",)

# The replica count the report's headline tables use.
DEFAULT_REPLICA_COUNT = 100

# Drawn at one IEEE column wide and included at \columnwidth, so the paper never
# rescales it: these two figures sit in the running text next to the paragraph
# that reads them off, not in a page-spanning float.
FIGURE_WIDTH = figures.SINGLE_COLUMN_WIDTH

# Matches thesis_tables.REPORTED_METRICS: the metrics the paper prints, in the
# same reading order.
METRICS = (
    ("adjacent_pair_acc", "Adjacent\nPair Accuracy"),
    ("exact_match", "Exact\nMatch"),
    ("longest_correct_run", "Longest\nCorrect Run"),
    ("edit_similarity", "Edit\nSimilarity"),
)

# The two arms these figures exist to compare, in ladder order.
PAIR = (("control", "Random Search"), ("agentic", "LLM-Guided"))

SPECIES = {"ecoli": "E. coli", "yeast": "S. cerevisiae"}


def _organism_key(run: Run) -> str:
    return (run.config.get("data") or {}).get("organism") or run.organism


def arm_mean(run: Run, arm: str, metric: str) -> float:
    return nanmean([(s.get(ARMS[arm]) or {}).get(metric) for s in run.samples])


def resolve_runs(explicit: list[str], results_root: Path, replica_count: int) -> list[Run]:
    """Explicit run folders if given, else one run per organism at
    ``replica_count``. A run without a control arm has nothing to compare
    against and is dropped."""
    if explicit:
        paths = [Path(v) if Path(v).exists() else results_root / v for v in explicit]
        runs = [load_run(path) for path in paths]
    else:
        by_organism: dict = {}
        for path in discover_runs(results_root):
            run = load_run(path)
            if run.replica_count == replica_count:
                by_organism[_organism_key(run)] = run  # later folder wins
        runs = list(by_organism.values())
    return sorted((r for r in runs if r.has_control), key=_organism_key)


def build_figure(run: Run, out_dir: Path, name: str) -> tuple[dict, list]:
    series = [
        (label, [arm_mean(run, arm, metric) for metric, _ in METRICS])
        for arm, label in PAIR
    ]
    written = figures.grouped_bars(
        [label for _, label in METRICS],
        series,
        f"Score ({figures.italic_species(SPECIES.get(_organism_key(run), run.organism))}, "
        f"{run.replica_count} replicas)",
        out_dir,
        name,
        FIGURE_FORMATS,
        width=FIGURE_WIDTH,
    )
    return written, series


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.search_comparison",
        description=(
            "Bar chart comparing the Random Search and LLM-Guided arms across the "
            "reported metrics, one figure per organism."
        ),
    )
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument(
        "--replica-count", type=int, default=DEFAULT_REPLICA_COUNT,
        help=f"replica count to auto-select runs at (default {DEFAULT_REPLICA_COUNT})",
    )
    parser.add_argument("--figure-out", default=str(DEFAULT_FIGURE_OUT))
    parser.add_argument(
        "--prefix", default="search_comparison",
        help="output filenames are <prefix>_<organism>.png",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    runs = resolve_runs(args.run, Path(args.results_root), args.replica_count)
    if not runs:
        print("No runs with a control arm found — pass --run explicitly.")
        return 1
    if not figures.available():
        print("matplotlib unavailable — nothing written.")
        return 1

    for run in runs:
        name = f"{args.prefix}_{_organism_key(run)}"
        written, series = build_figure(run, Path(args.figure_out), name)
        if args.quiet:
            continue
        print(f"\n{SPECIES.get(_organism_key(run), run.organism)} "
              f"r{run.replica_count} ({run.name}, n={run.n})")
        for index, (metric, _) in enumerate(METRICS):
            control, agentic = series[0][1][index], series[1][1][index]
            print(
                f"  {METRIC_NAMES[metric]:24s} Random Search {control:.3f}   "
                f"LLM-Guided {agentic:.3f}   delta {agentic - control:+.3f}"
            )
        for path in written.values():
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
