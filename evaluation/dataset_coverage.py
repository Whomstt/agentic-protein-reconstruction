"""Descriptive comparison of the evaluated 100-protein sample against the full
per-organism protein pool.

Answers one question: is the 100-protein sample typical of the pool it was drawn
from? Reports protein length, fragments per protein and fragment length for the
pool and for the sample, plus the fragment-count distribution as a figure.

One sample per organism, named in ``EVALUATED_RUN``.

Two bases are kept apart on purpose:

- Pool basis (the tables and the figure). Both groups -- pool and sample --
  are measured on the frozen replica-0 digestion in ``data/_frozen``, looked up
  by target sequence. Same digestion for both, so any difference is sampling,
  not digestion RNG.
- As-run basis (one extra table). Fragments per protein as the run actually
  saw them, recovered from the stored ordering length in ``samples.jsonl``.
  That dataset was digested separately, so these counts differ from the pool
  basis by digestion randomness as well as by sampling.

No model loading and no network: everything derives from the frozen JSONL pools
and the committed run outputs.

    python -m evaluation.dataset_coverage
    python -m evaluation.dataset_coverage --results-root final_results --out results/_analysis/coverage
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from evaluation import figures
from evaluation.exports import MIDRULE, Raw, Table, fmt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_TABLES = PROJECT_ROOT / "report" / "tables"
REPORT_IMAGES = PROJECT_ROOT / "report" / "images"
REPORT_FIGURES = PROJECT_ROOT / "report" / "figures"
FIGURE_NAME = "dataset_coverage"
SAMPLE_FIGURE_NAME = "sample_fragment_counts"

POOL_FILES = {
    "ecoli": PROJECT_ROOT / "data/_frozen/fragmented_ecoli.jsonl",
    "yeast": PROJECT_ROOT / "data/_frozen/fragmented_yeast.jsonl",
}
ORGANISM_LABEL = {"ecoli": "E. coli", "yeast": "S. cerevisiae"}
# The run whose 100 evaluated proteins are the reported sample, one per organism.
# Each run drew its own 100, so exactly one is used and it is reported simply as
# "the evaluated sample" rather than by replica count.
EVALUATED_RUN = {"ecoli": "ecoli_r100", "yeast": "yeast_r100"}


# --- stats (stdlib only, matching the rest of evaluation/) ---


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _describe(label: str, records: list[dict]) -> dict:
    """One table row: n, protein length, fragments/protein, fragment length.

    Fragment length is the mean over all fragments in the group (fragments tile
    the protein, so it is total residues / total fragments), with the median
    taken over individual fragments rather than over proteins."""
    lengths = [r["length"] for r in records]
    counts = [r["num_fragments"] for r in records]
    frag_lengths = [fl for r in records for fl in r["fragment_lengths"]]
    return {
        "group": label,
        "n_proteins": len(records),
        "seq_len_mean": _mean(lengths),
        "seq_len_median": _median(lengths),
        "seq_len_sd": _sd(lengths),
        "seq_len_min": min(lengths) if lengths else float("nan"),
        "seq_len_max": max(lengths) if lengths else float("nan"),
        "frags_per_protein_mean": _mean(counts),
        "frags_per_protein_median": _median(counts),
        "frags_per_protein_sd": _sd(counts),
        "frag_len_mean": _mean(frag_lengths),
        "frag_len_median": _median(frag_lengths),
        "frag_len_sd": _sd(frag_lengths),
    }


# --- loading ---


def load_pool(organism: str) -> dict[str, dict]:
    """Frozen pool keyed by target sequence. ``fragment_samples[0]`` is the
    replica-0 digestion, the fragment set the pipeline orders; the top-level
    ``fragments`` key is the union over replicas and is not used here."""
    pool: dict[str, dict] = {}
    with POOL_FILES[organism].open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            target = rec["target_reconstruction"]
            frags = rec["fragment_samples"][0]
            pool[target] = {
                "length": len(target),
                "num_fragments": len(frags),
                "fragment_lengths": [len(fr) for fr in frags],
            }
    return pool


def load_run_samples(run_dir: Path) -> list[dict]:
    """Per-sample target and as-run fragment count from a committed run."""
    path = run_dir / "samples.jsonl.gz"
    opener = gzip.open if path.exists() else None
    if opener is None:
        path = run_dir / "samples.jsonl"
        opener = open
    out = []
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            order = rec["order"]
            if isinstance(order, str):
                order = json.loads(order.replace("'", '"'))
            out.append({"target": rec["target"], "as_run_num_fragments": len(order)})
    return out


# --- reporting ---


def _fmt(value) -> str:
    if isinstance(value, (str, int)):
        return str(value)
    if value != value:  # NaN
        return "n/a"
    return f"{value:.1f}"


def _print_table(title: str, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    print(f"\n{title}")
    header = [head for _, head in columns]
    body = [[_fmt(row[key]) for key, _ in columns] for row in rows]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in body)) for i in range(len(columns))
    ]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in body:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))


POOL_COLUMNS = [
    ("group", "group"),
    ("n_proteins", "proteins"),
    ("seq_len_mean", "seq len mean"),
    ("seq_len_median", "median"),
    ("seq_len_sd", "sd"),
    ("seq_len_min", "min"),
    ("seq_len_max", "max"),
    ("frags_per_protein_mean", "frags/prot mean"),
    ("frags_per_protein_median", "median"),
    ("frag_len_mean", "frag len mean"),
    ("frag_len_median", "median"),
]


def build(results_root: Path) -> tuple[list[dict], list[dict], dict]:
    """Returns (pool-basis rows, as-run rows, per-organism fragment-count series
    for the figure)."""
    pool_rows: list[dict] = []
    as_run_rows: list[dict] = []
    series: dict[str, dict] = {}

    for organism, label in ORGANISM_LABEL.items():
        pool = load_pool(organism)
        pool_records = list(pool.values())
        pool_rows.append(_describe(f"{label} - full pool", pool_records))

        run_dir = results_root / EVALUATED_RUN[organism]
        if not run_dir.exists():
            raise SystemExit(f"{run_dir} not found")
        samples = load_run_samples(run_dir)
        missing = [s["target"] for s in samples if s["target"] not in pool]
        if missing:
            raise SystemExit(
                f"{run_dir}: {len(missing)} sampled proteins absent from the "
                f"{organism} frozen pool - pool and run are out of step."
            )
        records = [pool[s["target"]] for s in samples]
        group = f"{label} - evaluated sample (n={len(samples)})"
        pool_rows.append(_describe(group, records))

        as_run_counts = [s["as_run_num_fragments"] for s in samples]
        as_run_rows.append(
            {
                "group": group,
                "n_proteins": len(samples),
                "frags_per_protein_mean": _mean(as_run_counts),
                "frags_per_protein_median": _median(as_run_counts),
                "frags_per_protein_sd": _sd(as_run_counts),
                "frags_per_protein_min": min(as_run_counts),
                "frags_per_protein_max": max(as_run_counts),
            }
        )

        n_unique = len({s["target"] for s in samples})
        series[organism] = {
            "label": label,
            "pool": [r["num_fragments"] for r in pool_records],
            "sample": [r["num_fragments"] for r in records],
            "sample_as_run": as_run_counts,
            "coverage_pct": 100.0 * n_unique / len(pool),
            "n_pool": len(pool),
            "n_unique_sampled": n_unique,
        }

    return pool_rows, as_run_rows, series


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report_table(pool_rows: list[dict], series: dict) -> Table:
    """The report's booktabs table: full dataset against evaluated split, both
    organisms.

    Deliberately four numbers per row - protein count, mean protein length, mean
    fragments per protein, mean fragment length. Medians and SDs live in
    ``coverage_pool_basis.csv``; nine columns of them overran the IEEE text width
    and buried the one thing the table is for, which is that the split looks like
    the dataset."""
    rows: list = []
    for entry in series.values():
        label = entry["label"]
        group_rows = [r for r in pool_rows if r["group"].startswith(label)]
        if rows:
            rows.append(MIDRULE)
        for row in group_rows:
            name = "Full Dataset" if "full pool" in row["group"] else "Evaluated Split"
            rows.append(
                [
                    # The binomial has to be italic wherever it appears, so the
                    # cell opts out of escaping via Raw and keeps a plain form
                    # for the markdown/CSV views.
                    Raw(rf"\textit{{{label}}} - {name}", f"{label} - {name}"),
                    fmt(row["n_proteins"]),
                    fmt(float(row["seq_len_mean"]), 1),
                    fmt(float(row["frags_per_protein_mean"]), 1),
                    fmt(float(row["frag_len_mean"]), 1),
                ]
            )

    return Table(
        key="dataset_coverage",
        headers=[
            "Group",
            "P. Count",
            "P. Length Mean",
            "F. Count Mean",
            "F. Length Mean",
        ],
        rows=rows,
        column_spec="lrrrr",
        environment="table*",
        placement="!tb",
        raw_latex=True,
        caption=(
            "The 100 evaluated proteins against the full per-organism dataset that "
            "they were randomly drawn from. Protein length and fragment length are "
            "in residues."
        ),
        label="tab:dataset_coverage",
        notes=(
            "P.\\ = protein, F.\\ = fragment; lengths are in amino acids (aa). "
            "F.\\ Count Mean is the mean number of fragments a protein digests into. "
            "Both groups are measured on the same digestion replica, so any "
            "difference is sampling rather than digestion randomness."
        ),
    )


FIGURE_TEX = r"""\begin{{figure*}}[!tb]
  \centering
  \includegraphics[width=\textwidth]{{images/{name}.png}}
  \caption{{Fragments per protein for the 100 evaluated proteins against the full
  pool, per organism. The evaluated samples track the pool's distribution.}}
  \label{{fig:{name}}}
\end{{figure*}}
"""


def plot_fragment_distributions(
    series: dict, out_dir: Path, name: str, formats: tuple = ("pdf", "png")
) -> dict:
    """Fragment-count distribution, pool vs evaluated sample, one panel per
    organism. Density (not counts) so a 5k pool and a 100-protein sample share
    an axis. Greyscale-safe: the pool is a filled histogram, the sample a step
    outline, per the report's figure house style."""
    if not figures.available():
        print("  [skip] matplotlib unavailable - figure not written")
        return {}
    plt = figures._pyplot()
    figures.apply_style()

    organisms = list(series.keys())
    fig, axes = plt.subplots(
        1, len(organisms), figsize=(figures.SINGLE_COLUMN_WIDTH * 2, 2.4), sharey=False
    )
    if len(organisms) == 1:
        axes = [axes]

    for ax, organism in zip(axes, organisms):
        entry = series[organism]
        pool = entry["pool"]
        # Common bins across pool and samples; clipped at the 99th pool
        # percentile so a handful of very long proteins do not flatten the plot.
        cap = sorted(pool)[int(0.99 * (len(pool) - 1))]
        n_bins = 15  # coarse enough that a 100-protein step outline reads as a shape
        bins = [i * cap / n_bins for i in range(n_bins + 1)]
        clip = lambda xs: [min(x, cap) for x in xs]

        ax.hist(
            clip(pool),
            bins=bins,
            density=True,
            color=figures.GREYS[3],
            edgecolor="none",
            label=f"pool (n={len(pool)})",
        )
        sample = entry["sample"]
        ax.hist(
            clip(sample),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.2,
            color=figures.GREYS[0],
            linestyle=figures.LINESTYLES[0],
            label=f"evaluated sample (n={len(sample)})",
        )
        ax.set_title(
            f"{figures.italic_species(entry['label'])}: "
            f"{entry['coverage_pct']:.1f}% of pool sampled"
        )
        ax.set_xlabel("fragments per protein")
        ax.set_ylabel("density")
        ax.legend(frameon=False)

    fig.tight_layout()
    return figures._save(fig, out_dir, name, formats)


SAMPLE_FIGURE_TEX = r"""\begin{{figure*}}[!tb]
  \centering
  \includegraphics[width=\textwidth]{{images/{name}.png}}
  \caption{{Fragments per protein across the 100 evaluated proteins, per organism.
  Counts are the fragment sets the pipeline actually ordered. Both distributions
  are right-skewed: most proteins digest into a few dozen fragments, with a tail
  of much harder ones.}}
  \label{{fig:{name}}}
\end{{figure*}}
"""


def plot_sample_fragment_counts(
    series: dict, out_dir: Path, name: str, formats: tuple = ("pdf", "png")
) -> dict:
    """Fragment counts of the evaluated proteins alone, one panel per organism.

    Plain counts on the y axis rather than density: with n=100 per panel the
    reader can read the bar heights as proteins. Uses the as-run fragment sets,
    the ones the pipeline actually ordered."""
    if not figures.available():
        print("  [skip] matplotlib unavailable - figure not written")
        return {}
    plt = figures._pyplot()
    figures.apply_style()

    organisms = list(series.keys())
    fig, axes = plt.subplots(
        1, len(organisms), figsize=(figures.SINGLE_COLUMN_WIDTH * 2, 2.4)
    )
    if len(organisms) == 1:
        axes = [axes]

    for ax, organism in zip(axes, organisms):
        entry = series[organism]
        counts = entry["sample_as_run"]
        ax.hist(
            counts,
            bins=15,
            color=figures.GREYS[3],
            edgecolor=figures.GREYS[0],
            linewidth=0.6,
        )
        median = _median(counts)
        ax.axvline(
            median,
            color=figures.GREYS[0],
            linestyle="--",
            linewidth=1.0,
            label=f"median {median:.0f}",
        )
        ax.set_title(f"{figures.italic_species(entry['label'])} (n={len(counts)})")
        ax.set_xlabel("fragments per protein")
        ax.set_ylabel("proteins")
        ax.legend(frameon=False)

    fig.tight_layout()
    return figures._save(fig, out_dir, name, formats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="final_results")
    parser.add_argument("--out", default="results/_analysis/coverage")
    parser.add_argument("--table-out", default=str(REPORT_TABLES))
    parser.add_argument("--figure-out", default=str(REPORT_IMAGES))
    parser.add_argument("--figure-tex-out", default=str(REPORT_FIGURES))
    parser.add_argument(
        "--no-report", action="store_true", help="CSVs and console output only"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    pool_rows, as_run_rows, series = build(Path(args.results_root))

    _print_table(
        "Pool basis - frozen replica-0 digestion for every group (residues, counts)",
        pool_rows,
        POOL_COLUMNS,
    )
    _print_table(
        "As-run basis - fragments per protein as the run actually saw them",
        as_run_rows,
        [
            ("group", "group"),
            ("n_proteins", "proteins"),
            ("frags_per_protein_mean", "mean"),
            ("frags_per_protein_median", "median"),
            ("frags_per_protein_sd", "sd"),
            ("frags_per_protein_min", "min"),
            ("frags_per_protein_max", "max"),
        ],
    )

    print("\nCoverage")
    for organism, entry in series.items():
        print(
            f"  {entry['label']}: {entry['n_unique_sampled']} proteins evaluated "
            f"/ {entry['n_pool']} in pool "
            f"= {entry['coverage_pct']:.1f}%"
        )

    write_csv(out_dir / "coverage_pool_basis.csv", pool_rows)
    write_csv(out_dir / "coverage_as_run.csv", as_run_rows)
    written = [out_dir / "coverage_pool_basis.csv", out_dir / "coverage_as_run.csv"]

    if not args.no_report:
        table = build_report_table(pool_rows, series)
        table_path = Path(args.table_out) / f"{table.key}.tex"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        # Camera-ready, matching thesis_tables: these are \input{} into the paper.
        table_path.write_text(table.to_latex(comments=False), encoding="utf-8")
        written.append(table_path)

        # PNG only, alongside the report's other generated figures.
        for name, plot, tex in (
            (FIGURE_NAME, plot_fragment_distributions, FIGURE_TEX),
            (SAMPLE_FIGURE_NAME, plot_sample_fragment_counts, SAMPLE_FIGURE_TEX),
        ):
            images = plot(series, Path(args.figure_out), name, ("png",))
            written += list(images.values())
            if images:
                tex_path = Path(args.figure_tex_out) / f"{name}.tex"
                tex_path.parent.mkdir(parents=True, exist_ok=True)
                tex_path.write_text(tex.format(name=name), encoding="utf-8")
                written.append(tex_path)

    print()
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
