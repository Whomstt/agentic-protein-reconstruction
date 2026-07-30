"""Publication figures for the evaluation reports.

House style, applied by ``apply_style()``:

- Light theme, vector PDF for LaTeX inclusion. The PNG twin exists only
  so the markdown report has something to preview.
- Greyscale-safe: series are distinguished by marker and linestyle first, colour
  is a grey ramp. Nothing is identifiable by hue alone.
- 3.5 in wide (single column), 8 pt minimum type including tick labels.
- Kendall tau axes span [-1, 1]: its negative half means a reversed ordering,
  and clipping at 0 would hide the reversal failures.

matplotlib is imported lazily and the module degrades to a no-op with a clear
message if it is missing, so the rest of the report still generates."""

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

    The grey ramp runs light-to-dark so the last series carries the most weight:
    series arrive in method-ladder order, which puts the arm under test last."""
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


def _save(fig, out_dir: Path, name: str, formats: tuple = ("pdf", "png")) -> dict:
    """Write the figure once per requested format.

    Both by default: the PDF is what LaTeX includes, the PNG only exists so the
    markdown report has a preview. A caller that is including the PNG directly can
    ask for that alone rather than leaving an unused PDF beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for extension in formats:
        path = out_dir / f"{name}.{extension}"
        fig.savefig(path, format=extension)
        written[extension] = path
    _pyplot().close(fig)
    return written


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fragment_stratification(
    per_bin_by_arm: dict, metric: str, metric_label: str, out_dir: Path, name: str
) -> dict:
    """Metric vs fragment-count bin, one series per arm. Bins with no samples are
    left as gaps rather than interpolated over."""
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


def metrics_by_fragment_bin(
    panels: list, out_dir: Path, name: str, formats: tuple = ("pdf", "png")
) -> dict:
    """Every reported metric against fragment count, one panel per organism.

    Binned means rather than a raw scatter because the three metrics share one
    axis here: Exact Match is binary, so its per-protein points would be two rows
    of dots while the others formed clouds, and a bin mean reads as a rate for the
    binary metric and as an average for the continuous ones without changing the
    axis.

    The x axis is a real numeric one, with each bin plotted at its midpoint and
    ticks at the bin edges. Equal-width bins are what make that legal: with the
    widening bins used elsewhere in the report, evenly spaced ticks would stand for
    unequal ranges and the slope of the curve would partly be an artifact of the
    binning.

    ``panels`` is [(panel_title, centres, [(metric_label, values)])], where a None
    value leaves a gap rather than interpolating over an empty bin. Bin sizes are
    not drawn; whether a thinly populated interval needs a caveat is a question for
    the caption, not something to hang off the axis."""
    plt = _pyplot()
    apply_style()

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(SINGLE_COLUMN_WIDTH * len(panels), DEFAULT_HEIGHT + 0.35),
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]

    for ax, (title, centres, series) in zip(axes, panels):
        for index, (metric_label, values) in enumerate(series):
            xs = [c for c, v in zip(centres, values) if v is not None]
            ys = [v for v in values if v is not None]
            ax.plot(xs, ys, label=metric_label, **_series_style(index, len(series)))

        width = (centres[1] - centres[0]) if len(centres) > 1 else 1.0
        ax.set_xticks([c - width / 2 for c in centres] + [centres[-1] + width / 2])
        ax.set_xlim(centres[0] - width, centres[-1] + width)
        ax.set_xlabel("Fragments per protein")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title, pad=3)
        ax.set_ylabel("Score (LLM-Guided arm)")
        # Shared y limits, but every panel keeps its own numbers: these panels are
        # read one at a time as much as against each other, and a reader should
        # never have to track a value back across a gap to an unlabelled axis.
        ax.tick_params(labelleft=True)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 0.055),
        ncol=len(labels_), handlelength=2.6, columnspacing=1.6,
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.22)  # room for every panel's y labels
    return _save(fig, out_dir, name, formats)


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
    """Metric vs replica count, one series per organism. Replica count controls how
    many adjacencies the overlap graph can confirm, so this shows whether more
    digestion replicas buy real reconstruction quality."""
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
