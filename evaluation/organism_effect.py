"""Is E. coli easier than yeast for a reason other than protein size?

The fragment-correlation figure already shows quality falling as fragment count
rises, and E. coli proteins are smaller than yeast ones, so the raw E. coli minus
yeast gap confounds organism with size. This module reports three deliberately
plain numbers per replica count:

1. Spearman rho of each size predictor against each metric, per organism, plus a
   median split (mean score on the small half minus the large half) so the
   correlation has a units-of-the-metric companion.
2. The raw organism gap: mean(E. coli) - mean(yeast).
3. The size-adjusted organism gap: the same difference computed inside fragment
   count bins and then averaged over the pooled bin sizes (direct
   standardisation), so both organisms are compared at the same protein size.
   Significance comes from a label permutation test that shuffles organism
   within each bin, which needs no distributional assumption.

Also reported is the common-language effect size: the probability that a
randomly drawn E. coli protein scores above a randomly drawn yeast protein of
comparable fragment count.

No GPU, no model loading, no network - everything derives from samples.jsonl.

    python -m evaluation.organism_effect --results-root final_results
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

from evaluation.analysis import RESULTS_ROOT, Run, discover_runs, load_run
from evaluation.exports import MIDRULE, Table
from evaluation.stats import spearman

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLE_OUT = PROJECT_ROOT / "report" / "tables"

# The shipped arm. Correlating the shuffled floor would only re-describe chance.
ARM_KEY = "recon_metrics"
ARM_LABEL = "LLM-Guided"

METRICS = (
    ("adjacent_pair_acc", "Adjacent Pair Accuracy"),
    ("exact_match", "Exact Match"),
    ("longest_correct_run", "Longest Correct Run"),
    ("edit_similarity", "Edit Similarity"),
)

PREDICTORS = (
    ("num_fragments", "Fragments per protein"),
    ("mean_fragment_length", "Mean fragment length"),
    ("protein_length", "Protein length (residues)"),
)

SPECIES = {"ecoli": "E. coli", "yeast": "S. cerevisiae"}

# Bin width for the size-adjusted comparison, in fragments. Wide enough that most
# bins hold both organisms, narrow enough that "same size" means something.
BIN_WIDTH = 10

# A bin contributes to the adjusted gap only if it has at least this many
# proteins from each organism; otherwise a single sample would set a bin mean.
MIN_PER_ARM = 3

PERMUTATIONS = 10000
SEED = 20260805


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and value == value


def rows_for(run: Run) -> list[dict]:
    """One row per protein: size predictors plus the reported metrics."""
    out = []
    for sample in run.samples:
        n_fragments = len(sample.get("order") or [])
        target = sample.get("target") or ""
        if not n_fragments or not target:
            continue
        metrics = sample.get(ARM_KEY) or {}
        row = {
            "num_fragments": n_fragments,
            "mean_fragment_length": len(target) / n_fragments,
            "protein_length": len(target),
        }
        for key, _ in METRICS:
            value = metrics.get(key)
            row[key] = float(value) if _is_number(value) else None
        out.append(row)
    return out


def _paired(rows: list[dict], a: str, b: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in rows:
        if row.get(a) is not None and row.get(b) is not None:
            xs.append(row[a])
            ys.append(row[b])
    return xs, ys


def median_split(rows: list[dict], predictor: str, metric: str) -> dict:
    """Mean metric on the small half of proteins minus the large half.

    Same story as the correlation, told in the metric's own units: "proteins in
    the smaller half score X higher"."""
    xs, ys = _paired(rows, predictor, metric)
    if len(xs) < 4:
        return {}
    cut = statistics.median(xs)
    small = [y for x, y in zip(xs, ys) if x <= cut]
    large = [y for x, y in zip(xs, ys) if x > cut]
    if not small or not large:
        return {}
    return {
        "cut": cut,
        "small_mean": statistics.fmean(small),
        "large_mean": statistics.fmean(large),
        "delta": statistics.fmean(small) - statistics.fmean(large),
        "n_small": len(small),
        "n_large": len(large),
    }


def _bin_index(n_fragments: float) -> int:
    return int(n_fragments) // BIN_WIDTH


def _bin_label(index: int) -> str:
    return f"{index * BIN_WIDTH + 1}-{(index + 1) * BIN_WIDTH}"


def _stratified_delta(
    pairs: list[tuple[int, str, float]],
) -> tuple[float, float, list[dict]]:
    """Pooled-size-weighted mean difference across fragment-count bins.

    ``pairs`` is (bin index, organism key, metric value). Bins without enough of
    both organisms are dropped, so the weights come from the bins that survive.
    Returns the difference, the yeast mean over the same bins and weights (the
    baseline for a relative gap), and the per-bin detail."""
    by_bin: dict[int, dict[str, list[float]]] = {}
    for index, organism, value in pairs:
        by_bin.setdefault(index, {}).setdefault(organism, []).append(value)

    detail, weighted, base, total = [], 0.0, 0.0, 0
    for index in sorted(by_bin):
        arms = by_bin[index]
        if len(arms) != 2 or any(len(v) < MIN_PER_ARM for v in arms.values()):
            continue
        ecoli, yeast = arms["ecoli"], arms["yeast"]
        delta = statistics.fmean(ecoli) - statistics.fmean(yeast)
        weight = len(ecoli) + len(yeast)
        weighted += delta * weight
        # The yeast mean over the same bins and weights, so the relative gap
        # divides by the baseline the adjusted difference was measured against
        # rather than by the unmatched yeast average.
        base += statistics.fmean(yeast) * weight
        total += weight
        detail.append({
            "bin": _bin_label(index),
            "delta": delta,
            "ecoli_mean": statistics.fmean(ecoli),
            "yeast_mean": statistics.fmean(yeast),
            "n_ecoli": len(ecoli),
            "n_yeast": len(yeast),
        })
    nan = float("nan")
    return (
        weighted / total if total else nan,
        base / total if total else nan,
        detail,
    )


def _permutation_p(pairs: list[tuple[int, str, float]], observed: float) -> float:
    """Two-sided p for the stratified gap, shuffling organism labels within bins.

    Within-bin shuffling is what makes this a size-adjusted test: it destroys the
    organism signal while leaving the size distribution of each bin untouched."""
    if observed != observed:
        return float("nan")
    rng = random.Random(SEED)
    by_bin: dict[int, list[tuple[str, float]]] = {}
    for index, organism, value in pairs:
        by_bin.setdefault(index, []).append((organism, value))

    extreme = 0
    for _ in range(PERMUTATIONS):
        shuffled = []
        for index, members in by_bin.items():
            labels = [organism for organism, _ in members]
            rng.shuffle(labels)
            shuffled += [
                (index, label, value)
                for label, (_, value) in zip(labels, members)
            ]
        delta, _, _ = _stratified_delta(shuffled)
        if delta == delta and abs(delta) >= abs(observed) - 1e-12:
            extreme += 1
    # +1 in both places: the observed labelling is itself one of the outcomes, so
    # the p value can never be reported as exactly zero.
    return (extreme + 1) / (PERMUTATIONS + 1)


def _common_language(pairs: list[tuple[int, str, float]]) -> float:
    """P(random E. coli protein > random yeast protein), within the same bin.

    Ties count a half, as in the usual common-language effect size."""
    by_bin: dict[int, dict[str, list[float]]] = {}
    for index, organism, value in pairs:
        by_bin.setdefault(index, {}).setdefault(organism, []).append(value)

    wins, total = 0.0, 0
    for arms in by_bin.values():
        if len(arms) != 2 or any(len(v) < MIN_PER_ARM for v in arms.values()):
            continue
        for a in arms["ecoli"]:
            for b in arms["yeast"]:
                wins += 1.0 if a > b else 0.5 if a == b else 0.0
                total += 1
    return wins / total if total else float("nan")


def organism_effect(by_organism: dict[str, list[dict]], metric: str) -> dict:
    ecoli = [r[metric] for r in by_organism.get("ecoli", []) if r.get(metric) is not None]
    yeast = [r[metric] for r in by_organism.get("yeast", []) if r.get(metric) is not None]
    if not ecoli or not yeast:
        return {}

    pairs = [
        (_bin_index(row["num_fragments"]), organism, row[metric])
        for organism, rows in by_organism.items()
        for row in rows
        if row.get(metric) is not None
    ]
    adjusted, adjusted_base, detail = _stratified_delta(pairs)

    def relative(delta: float, base: float) -> float:
        """Delta as a percentage of the yeast baseline it was measured against."""
        return delta / base * 100 if base else float("nan")

    ecoli_mean, yeast_mean = statistics.fmean(ecoli), statistics.fmean(yeast)
    return {
        "ecoli_mean": ecoli_mean,
        "yeast_mean": yeast_mean,
        "raw_delta": ecoli_mean - yeast_mean,
        "adjusted_delta": adjusted,
        "raw_pct": relative(ecoli_mean - yeast_mean, yeast_mean),
        "adjusted_pct": relative(adjusted, adjusted_base),
        "p_value": _permutation_p(pairs, adjusted),
        "common_language": _common_language(pairs),
        "bins": detail,
        "n_ecoli": len(ecoli),
        "n_yeast": len(yeast),
    }


def _fmt(value, places: int = 3) -> str:
    return "n/a" if value is None or value != value else f"{value:.{places}f}"


def collect(runs: list[Run]) -> dict[int, dict[str, list[dict]]]:
    """{replica count: {organism: per-protein rows}} across every run."""
    by_replica: dict[int, dict[str, list[dict]]] = {}
    for run in runs:
        organism = (run.config.get("data") or {}).get("organism") or run.organism
        by_replica.setdefault(run.replica_count, {})[organism] = rows_for(run)
    return by_replica


def build_table(by_replica: dict[int, dict[str, list[dict]]]) -> Table:
    """The organism gap, before and after matching on fragment count.

    Means stay in the metrics' native 0-1 scale; both gaps are relative, as a
    percentage of the yeast mean. An absolute difference understates a gap on a
    low-scoring metric - Exact Match at 0.460 vs 0.260 is +20 points but nearly
    twice the score - and the metrics here span very different bases, so a
    relative gap is the one that stays comparable down a column."""
    rows: list = []
    for replicas in sorted(by_replica, reverse=True):
        by_organism = by_replica[replicas]
        if rows:
            rows.append(MIDRULE)  # one rule per replica block
        for position, (metric, metric_label) in enumerate(METRICS):
            effect = organism_effect(by_organism, metric)
            if not effect:
                continue
            rows.append([
                str(replicas) if position == 0 else "",
                metric_label,
                f"{effect['ecoli_mean']:.3f}",
                f"{effect['yeast_mean']:.3f}",
                f"{effect['raw_pct']:+.1f}",
                f"{effect['adjusted_pct']:+.1f}",
            ])

    return Table(
        key="organism_gap",
        headers=[
            "Replicas",
            "Metric",
            r"\textit{E. coli}",
            r"\textit{S. cerevisiae}",
            r"Raw gap (\%)",
            r"Size-adj.\ gap (\%)",
        ],
        rows=rows,
        column_spec="rlrrrr",
        environment="table*",
        placement="!t",
        raw_latex=True,
        # No \textit here: IEEEtran sets table captions in small caps, which has
        # no italic shape, so a \textit inside one warns "OT1/ptm/m/scit
        # undefined". The species names stay italic in the column headers, which
        # are typeset in the normal font.
        caption=(
            "Organism gap for the LLM-Guided pipeline, before and after matching "
            "on protein size."
        ),
        label="tab:organism_gap",
        notes=(
            "Both gaps say how much higher E. coli scores than S. cerevisiae, as "
            "a percentage of the S. cerevisiae mean: $+76.9$ means an E. coli "
            "score nearly twice the yeast one. The size-adjusted gap is the same "
            f"comparison made within bins of {BIN_WIDTH} fragments, so the two "
            "organisms are matched on protein size. It collapses towards zero at "
            "20 and 100 replicas and changes sign at 5, and no cell is "
            f"significant (two-sided permutation test, {PERMUTATIONS} shuffles), "
            "so the raw gap is mostly a size effect. "
            "Means are per-protein averages over 100 proteins on each metric's "
            "native 0--1 scale."
        ),
    )


def report(runs: list[Run]) -> list[str]:
    lines: list[str] = []
    add = lines.append

    by_replica = collect(runs)

    for replicas in sorted(by_replica, reverse=True):
        by_organism = by_replica[replicas]
        add("")
        add("=" * 78)
        add(f"{replicas} digestion replicas — {ARM_LABEL} arm")
        add("=" * 78)

        add("")
        add("A. Does protein size predict reconstruction quality?")
        add("")
        header = (
            f"  {'organism':<14}{'predictor':<26}{'metric':<24}"
            f"{'rho':>8}{'p':>10}{'small-large':>13}{'n':>6}"
        )
        add(header)
        add("  " + "-" * (len(header) - 2))
        for organism, rows in sorted(by_organism.items()):
            for predictor, predictor_label in PREDICTORS:
                for metric, metric_label in METRICS:
                    xs, ys = _paired(rows, predictor, metric)
                    result = spearman(xs, ys)
                    split = median_split(rows, predictor, metric)
                    add(
                        f"  {SPECIES.get(organism, organism):<14}{predictor_label:<26}"
                        f"{metric_label:<24}{result.statistic:>+8.3f}"
                        f"{result.pvalue:>10.4f}"
                        f"{_fmt(split.get('delta')):>13}{result.n:>6}"
                    )

        add("")
        add("B. E. coli vs yeast, before and after matching on fragment count")
        add("")
        header = (
            f"  {'metric':<24}{'E. coli':>9}{'yeast':>9}{'raw gap':>10}"
            f"{'adj. gap':>10}{'perm p':>9}{'P(E>Y)':>9}"
        )
        add(header)
        add("  " + "-" * (len(header) - 2))
        for metric, metric_label in METRICS:
            effect = organism_effect(by_organism, metric)
            if not effect:
                continue
            add(
                f"  {metric_label:<24}{effect['ecoli_mean']:>9.3f}"
                f"{effect['yeast_mean']:>9.3f}{effect['raw_delta']:>+10.3f}"
                f"{effect['adjusted_delta']:>+10.3f}{effect['p_value']:>9.4f}"
                f"{effect['common_language']:>9.3f}"
            )

        add("")
        add(
            f"  Bin-by-bin detail for {METRICS[0][1]} "
            f"(bins of {BIN_WIDTH} fragments, >= {MIN_PER_ARM} proteins per organism):"
        )
        add("")
        detail = organism_effect(by_organism, METRICS[0][0]).get("bins") or []
        if not detail:
            add("    no fragment-count bin holds enough of both organisms")
        else:
            add(
                f"    {'fragments':<12}{'E. coli':>9}{'n':>5}"
                f"{'yeast':>9}{'n':>5}{'gap':>9}"
            )
            for row in detail:
                add(
                    f"    {row['bin']:<12}{row['ecoli_mean']:>9.3f}{row['n_ecoli']:>5}"
                    f"{row['yeast_mean']:>9.3f}{row['n_yeast']:>5}{row['delta']:>+9.3f}"
                )

    add("")
    add("Reading these numbers")
    add("-" * 78)
    add(
        "  rho          Spearman rank correlation. Negative means quality falls as the"
    )
    add(
        "               predictor rises. |rho| ~0.1 weak, ~0.3 moderate, ~0.5 strong."
    )
    add(
        "  small-large  Mean metric on the smaller half of proteins minus the larger"
    )
    add("               half, split at the median. Same story in the metric's units.")
    add("  raw gap      mean(E. coli) - mean(yeast), unmatched.")
    add(
        "  adj. gap     The same difference computed within fragment-count bins and"
    )
    add(
        "               re-averaged over the pooled bin sizes. What is left after size."
    )
    add(
        f"  perm p       Two-sided, {PERMUTATIONS} shuffles of organism within bin."
    )
    add(
        "  P(E>Y)       Chance a random E. coli protein beats a random yeast protein of"
    )
    add("               similar fragment count. 0.5 is no organism effect.")
    add("")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.organism_effect",
        description=(
            "Size-vs-quality correlations and the E. coli / yeast gap before and "
            "after matching on fragment count."
        ),
    )
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument(
        "--replica-count", type=int, action="append", default=[],
        help="restrict to these replica counts (repeatable); default is all",
    )
    parser.add_argument("--out", default="", help="also write the report to this file")
    parser.add_argument("--table-out", default=str(DEFAULT_TABLE_OUT))
    args = parser.parse_args(argv)

    runs = [load_run(path) for path in discover_runs(Path(args.results_root))]
    if args.replica_count:
        runs = [run for run in runs if run.replica_count in args.replica_count]
    if not runs:
        print("No runs found — check --results-root.")
        return 1

    lines = report(runs)
    text = "\n".join(lines)
    print(text)

    table = build_table(collect(runs))
    table_path = Path(args.table_out) / f"{table.key}.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    # Camera-ready, matching thesis_tables: these are \input{} into the paper.
    table_path.write_text(table.to_latex(comments=False), encoding="utf-8")
    print(f"wrote {table_path}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
