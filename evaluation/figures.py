"""Publication figures for the evaluation reports.

House style, applied by ``apply_style()`` and relied on by every figure here:

- **Light theme, vector PDF.** PDF is the artifact to ``\\includegraphics{}``.
  A PNG twin is written alongside purely so the markdown report has something
  to preview; the PDF is the one that goes in the dissertation.
- **Greyscale-safe.** Series are distinguished by MARKER and LINESTYLE first;
  colour is a grey ramp that survives a black-and-white printer. Nothing in a
  figure is identifiable by hue alone.
- **3.5 in wide** (single column) and **8 pt minimum** type anywhere, including
  tick labels, so nothing becomes unreadable after the figure is placed.
- **Kendall tau axes span [-1, 1]**, not [0, 1]: tau is a correlation whose
  negative half is meaningful (a reversed ordering), and clipping it at 0 hides
  the reversal failures the error taxonomy is about.

matplotlib is imported lazily and the module degrades to a no-op with a clear
message if it is missing, so the rest of the report still generates.
"""

from __future__ import annotations

from pathlib import Path

SINGLE_COLUMN_WIDTH = 3.5  # inches
DEFAULT_HEIGHT = 2.4
MIN_FONT_PT = 8

# Grey ramp: distinct in colour AND when printed greyscale.
GREYS = ["#000000", "#4d4d4d", "#808080", "#a6a6a6", "#cccccc"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

# Metrics whose natural axis is [-1, 1] rather than [0, 1].
SIGNED_METRICS = {"kendall_tau"}


class FiguresUnavailable(RuntimeError):
    pass


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless; never opens a window
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover
        raise FiguresUnavailable(
            "matplotlib is not installed — figures skipped. Install with: "
            "python -m pip install matplotlib"
        ) from exc


def available() -> bool:
    try:
        _pyplot()
        return True
    except FiguresUnavailable:
        return False


def apply_style() -> None:
    plt = _pyplot()
    plt.rcParams.update(
        {
            "figure.figsize": (SINGLE_COLUMN_WIDTH, DEFAULT_HEIGHT),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": MIN_FONT_PT,
            "axes.labelsize": MIN_FONT_PT,
            "axes.titlesize": MIN_FONT_PT + 1,
            "xtick.labelsize": MIN_FONT_PT,
            "ytick.labelsize": MIN_FONT_PT,
            "legend.fontsize": MIN_FONT_PT,
            "font.family": "sans-serif",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,  # embed TrueType, keeps text selectable/editable
            "ps.fonttype": 42,
        }
    )


def _series_style(index: int, total: int | None = None) -> dict:
    """Style for series ``index`` of ``total``.

    The grey ramp runs light-to-dark so the LAST series carries the most visual
    weight: series are passed in method-ladder order, which puts the arm under
    test last. Marker and linestyle still carry the identity on their own, so
    the figure survives greyscale printing regardless of the shading.
    """
    if total and total > 1:
        step = min(total, len(GREYS))
        color = GREYS[max(0, step - 1 - min(index, step - 1))]
    else:
        color = GREYS[index % len(GREYS)]
    return {
        "color": color,
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
        "markeredgecolor": "#000000",
        "markeredgewidth": 0.4,
    }


def _apply_metric_axis(ax, metric: str) -> None:
    """Kendall tau gets the full [-1, 1]; the others [0, 1] with headroom."""
    if metric in SIGNED_METRICS:
        ax.set_ylim(-1.0, 1.0)
        ax.axhline(0.0, color="#999999", linewidth=0.5, linestyle="-", zorder=1)
    else:
        ax.set_ylim(0.0, 1.0)


def _save(fig, out_dir: Path, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png")
    _pyplot().close(fig)
    return {"pdf": pdf, "png": png}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fragment_stratification(
    per_bin_by_arm: dict, metric: str, metric_label: str, out_dir: Path, name: str
) -> dict:
    """Metric vs fragment-count bin, one series per arm.

    The headline difficulty axis: performance against how many pieces the
    protein was cut into. Bins with no samples are left as gaps rather than
    interpolated over.
    """
    plt = _pyplot()
    apply_style()
    fig, ax = plt.subplots()

    labels = None
    for index, (arm_label, per_bin) in enumerate(per_bin_by_arm.items()):
        labels = list(per_bin)
        xs, ys = [], []
        for position, bin_label in enumerate(labels):
            values = per_bin[bin_label]["values"]
            if values:
                xs.append(position)
                ys.append(sum(values) / len(values))
        ax.plot(xs, ys, label=arm_label, **_series_style(index, len(per_bin_by_arm)))

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    ax.set_xlabel("Fragments per protein")
    ax.set_ylabel(metric_label)
    _apply_metric_axis(ax, metric)
    ax.legend(loc="best", handlelength=2.6)
    return _save(fig, out_dir, name)


def lift_by_bin(lift: dict, metric_label: str, out_dir: Path, name: str) -> dict:
    """Paired lift over the shuffled floor, per fragment-count bin.

    Bars are the per-sample mean difference (agentic - shuffled), so this is a
    paired quantity, not a gap between two independent means.
    """
    plt = _pyplot()
    apply_style()
    fig, ax = plt.subplots()

    labels = list(lift)
    values = [
        lift[label]["lift"] if lift[label]["n_usable"] else 0.0 for label in labels
    ]
    positions = range(len(labels))
    ax.bar(
        positions,
        values,
        color="#b0b0b0",
        edgecolor="#000000",
        linewidth=0.6,
        width=0.62,
        hatch="///",
    )
    ax.axhline(0.0, color="#000000", linewidth=0.6)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Fragments per protein")
    ax.set_ylabel(f"Lift in {metric_label}")
    for position, value in zip(positions, values):
        ax.annotate(
            f"{value:+.2f}",
            (position, value),
            textcoords="offset points",
            xytext=(0, 2 if value >= 0 else -9),
            ha="center",
            fontsize=MIN_FONT_PT,
        )
    return _save(fig, out_dir, name)


def method_ladder(records: list[dict], metric: str, metric_label: str, out_dir: Path, name: str) -> dict:
    """Arm means with confidence intervals — the method ladder for one metric.

    Error bars are the Wilson interval for Exact Match and the BCa bootstrap
    interval for the continuous metrics.
    """
    plt = _pyplot()
    apply_style()
    fig, ax = plt.subplots()

    labels = [r["arm_label"] for r in records]
    points = [r["point"] for r in records]
    lows = [max(0.0, r["point"] - r["low"]) if r["low"] is not None else 0.0 for r in records]
    highs = [max(0.0, r["high"] - r["point"]) if r["high"] is not None else 0.0 for r in records]

    positions = range(len(labels))
    ax.errorbar(
        list(positions),
        points,
        yerr=[lows, highs],
        fmt="o",
        color="#000000",
        ecolor="#666666",
        elinewidth=0.9,
        capsize=2.5,
        markersize=4,
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric_label)
    _apply_metric_axis(ax, metric)
    return _save(fig, out_dir, name)


def replica_scaling(
    series: dict, metric: str, metric_label: str, out_dir: Path, name: str
) -> dict:
    """Metric vs replica count, one series per organism.

    Replica count controls how many confirmed adjacencies the overlap graph can
    assert, so this is the figure that shows whether more digestion replicas buy
    real reconstruction quality.
    """
    plt = _pyplot()
    apply_style()
    fig, ax = plt.subplots()

    for index, (label, points) in enumerate(series.items()):
        ordered = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in ordered]
        ys = [p[1] for p in ordered]
        ax.plot(xs, ys, label=label, **_series_style(index, len(series)))

    ax.set_xscale("log")
    ax.set_xlabel("Digestion replicas")
    ax.set_ylabel(metric_label)
    all_x = sorted({p[0] for points in series.values() for p in points})
    if all_x:
        ax.set_xticks(all_x)
        ax.set_xticklabels([str(int(x)) for x in all_x])
        ax.minorticks_off()
    _apply_metric_axis(ax, metric)
    ax.legend(loc="best", handlelength=2.6)
    return _save(fig, out_dir, name)


def error_taxonomy(counts_by_run: dict, class_labels: dict, out_dir: Path, name: str) -> dict:
    """Stacked composition of failure shapes per run.

    Segments are ordered best-to-worst outcome and shaded along the grey ramp
    with distinct hatching, so the composition survives greyscale printing.
    """
    plt = _pyplot()
    apply_style()
    height = max(DEFAULT_HEIGHT, 0.42 * len(counts_by_run) + 1.1)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, height))

    run_labels = list(counts_by_run)
    ordered_classes = [
        key for key in class_labels if any(counts_by_run[r].get(key) for r in run_labels)
    ]
    hatches = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo"]

    lefts = [0.0] * len(run_labels)
    for index, key in enumerate(ordered_classes):
        widths = []
        for position, run_label in enumerate(run_labels):
            counts = counts_by_run[run_label]
            total = sum(counts.values()) or 1
            widths.append(100.0 * counts.get(key, 0) / total)
        ax.barh(
            range(len(run_labels)),
            widths,
            left=lefts,
            label=class_labels[key],
            color=GREYS[index % len(GREYS)] if index else "#ffffff",
            edgecolor="#000000",
            linewidth=0.5,
            hatch=hatches[index % len(hatches)],
            height=0.66,
        )
        lefts = [l + w for l, w in zip(lefts, widths)]

    ax.set_yticks(range(len(run_labels)))
    ax.set_yticklabels(run_labels)
    ax.set_xlabel("Share of samples (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        handlelength=1.8,
        borderaxespad=0.0,
    )
    return _save(fig, out_dir, name)


def paired_gain_distribution(deltas: list[float], xlabel: str, out_dir: Path, name: str) -> dict:
    """Histogram of per-sample paired gains, with the zero line marked.

    The paired distribution is what the Wilcoxon test actually sees; a mean
    delta alone hides whether a gain is broad or driven by a few proteins.
    """
    plt = _pyplot()
    apply_style()
    fig, ax = plt.subplots()

    usable = [d for d in deltas if d is not None]
    if usable:
        ax.hist(
            usable,
            bins=min(20, max(6, len(usable) // 5)),
            color="#b0b0b0",
            edgecolor="#000000",
            linewidth=0.5,
        )
    ax.axvline(0.0, color="#000000", linewidth=1.0, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Samples")
    return _save(fig, out_dir, name)
