from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend

from evaluation.metrics import METRIC_NAMES
from evaluation.reporting import (
    METRIC_RANGES,
    _aggregate_iteration_stats,
    _format_config_rows,
    _format_distribution_rows,
    _format_duration,
    _format_iteration_gain_distribution_rows,
    _format_iteration_gain_rows,
    _format_metric_rows,
    _format_per_sample_rows,
    _validity_best_so_far_series,
    validity_score_histogram_bins,
)

BASELINE_COLOR = colors.HexColor("#9aa5b1")
FIRST_PASS_COLOR = colors.HexColor("#f59e0b")
RECON_COLOR = colors.HexColor("#3b82f6")
GRID_COLOR = colors.HexColor("#e5e7eb")
HEADER_BG = colors.HexColor("#1f2937")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=6
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Note", parent=styles["Normal"], fontSize=9, textColor=colors.grey
        )
    )
    return styles


def _table(headers, rows, col_widths=None, font_size=8):
    data = [headers] + rows
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.5, GRID_COLOR),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _metric_bar_drawing(
    baseline_averages, recon_averages, first_pass_averages=None, width=500, height=260
):
    keys = list(METRIC_NAMES.keys())

    def norm(key, v):
        lo, hi = METRIC_RANGES[key]
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))

    baseline_vals = [norm(k, baseline_averages.get(k, 0.0)) for k in keys]
    recon_vals = [norm(k, recon_averages.get(k, 0.0)) for k in keys]

    drawing = Drawing(width, height)
    chart = VerticalBarChart()
    chart.x = 45
    chart.y = 55
    chart.width = width - 70
    chart.height = height - 90
    if first_pass_averages is not None:
        first_pass_vals = [norm(k, first_pass_averages.get(k, 0.0)) for k in keys]
        chart.data = [baseline_vals, first_pass_vals, recon_vals]
        chart.bars[0].fillColor = BASELINE_COLOR
        chart.bars[1].fillColor = FIRST_PASS_COLOR
        chart.bars[2].fillColor = RECON_COLOR
    else:
        chart.data = [baseline_vals, recon_vals]
        chart.bars[0].fillColor = BASELINE_COLOR
        chart.bars[1].fillColor = RECON_COLOR
    chart.categoryAxis.categoryNames = [
        METRIC_NAMES[k].replace(" ", "\n") for k in keys
    ]
    chart.categoryAxis.labels.fontSize = 6
    chart.categoryAxis.labels.dy = -10
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 1
    chart.valueAxis.labels.fontSize = 7
    chart.barSpacing = 2
    chart.groupSpacing = 8
    drawing.add(chart)

    legend = Legend()
    legend.x = width - 150
    legend.y = height - 10
    legend.dx = 8
    legend.dy = 8
    legend.fontSize = 8
    legend.columnMaximum = 1
    legend_entries = [(BASELINE_COLOR, "Shuffled baseline")]
    if first_pass_averages is not None:
        legend_entries.append((FIRST_PASS_COLOR, "1st iteration"))
    legend_entries.append((RECON_COLOR, "Best (iterative)"))
    legend.colorNamePairs = legend_entries
    drawing.add(legend)

    drawing.add(
        String(
            width / 2,
            height - 12,
            "Metric Comparison (normalized 0-1 scale; see table for raw values)",
            textAnchor="middle",
            fontSize=9,
        )
    )
    return drawing


def _validity_progression_drawing(sample_reports, width=500, height=260):
    from reportlab.graphics.charts.lineplots import LinePlot

    series = _validity_best_so_far_series(sample_reports)
    if not series:
        return None

    drawing = Drawing(width, height)
    chart = LinePlot()
    chart.x = 55
    chart.y = 40
    chart.width = width - 90
    chart.height = height - 70

    aggregate = len(series) > 12
    if aggregate:
        max_len, mean_line, _, _ = _aggregate_iteration_stats(series)
        points = [
            (idx + 1, v) for idx, v in enumerate(mean_line) if v is not None
        ]
        if not points:
            return None
        chart.data = [points]
        chart.lines[0].strokeColor = RECON_COLOR
        chart.lines[0].strokeWidth = 2
        title = f"Mean Best Validity Score by Iteration (n={len(series)} samples)"
    else:
        chart.data = [
            [(idx + 1, v) for idx, v in enumerate(s["values"]) if v is not None]
            for s in series
        ]
        palette = [
            colors.HexColor(c)
            for c in ("#3b82f6", "#f97316", "#10b981", "#e11d48", "#8b5cf6", "#eab308")
        ]
        # Index chart.lines explicitly, bounded by the number of series. Do NOT
        # iterate `chart.lines` directly: reportlab's TypedPropertyCollection
        # auto-creates and caches a style object for any index and never raises
        # IndexError, so `enumerate(chart.lines)` loops forever, allocating
        # millions of objects until the process OOMs.
        for i in range(len(chart.data)):
            line = chart.lines[i]
            line.strokeColor = palette[i % len(palette)]
            line.strokeWidth = 1.5
        title = "Best Validity Score by Iteration (per sample)"

    chart.xValueAxis.labels.fontSize = 7
    chart.yValueAxis.labels.fontSize = 7
    drawing.add(chart)
    drawing.add(
        String(width / 2, height - 12, title, textAnchor="middle", fontSize=9)
    )
    return drawing


def _validity_histogram_drawing(sample_reports, width=500, height=260):
    binned = validity_score_histogram_bins(sample_reports, bins=10)
    if binned is None:
        return None
    lo, hi, _, counts = binned

    drawing = Drawing(width, height)
    chart = VerticalBarChart()
    chart.x = 45
    chart.y = 40
    chart.width = width - 70
    chart.height = height - 70
    chart.data = [counts]
    chart.categoryAxis.categoryNames = [""] * len(counts)
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = RECON_COLOR
    chart.barSpacing = 1
    drawing.add(chart)
    drawing.add(
        String(
            width / 2,
            height - 12,
            f"Best Validity Score Distribution (n={sum(counts)}, range {lo:.2f}-{hi:.2f})",
            textAnchor="middle",
            fontSize=9,
        )
    )
    return drawing


def write_pdf_report(payload: dict, sample_reports: list[dict], run_dir: Path) -> Path:
    pdf_path = run_dir / "report.pdf"
    styles = _styles()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=payload.get("run_name", "Reconstruction Report"),
    )

    story = [Paragraph(payload.get("run_name", "Reconstruction Report"), styles["ReportTitle"])]

    duration = payload.get("duration_seconds")
    exact_matches = sum(
        1 for s in sample_reports if s.get("recon_metrics", {}).get("exact_match")
    )
    overview = [
        f"Samples evaluated: {payload.get('sample_count', len(sample_reports))}",
        f"Exact matches: {exact_matches}/{len(sample_reports)}",
        f"Avg junctions pruned: {payload.get('avg_pruned', 0.0):.1f}%",
        f"Result folder: {run_dir.name}",
    ]
    if duration is not None:
        overview.append(f"Total run duration: {_format_duration(duration)}")
        if sample_reports:
            overview.append(
                f"Avg duration per sample: {_format_duration(duration / len(sample_reports))}"
            )
    story.append(Paragraph("Run Overview", styles["SectionHeading"]))
    for line in overview:
        story.append(Paragraph(f"&bull; {line}", styles["Normal"]))

    story.append(Paragraph("Configuration", styles["SectionHeading"]))
    story.append(
        _table(
            ["Setting", "Value"],
            _format_config_rows(payload["config"]),
            col_widths=[6 * cm, 10 * cm],
        )
    )

    story.append(Paragraph("Benchmark Summary", styles["SectionHeading"]))
    story.append(
        _table(
            ["Metric", "Baseline", "Reconstructed", "Delta", "Interpretation", "Direction"],
            _format_metric_rows(payload),
            col_widths=[5.5 * cm, 2.3 * cm, 2.6 * cm, 2.3 * cm, 2.6 * cm, 1.7 * cm],
        )
    )

    bar_drawing = _metric_bar_drawing(
        payload.get("baseline_averages", {}),
        payload.get("recon_averages", {}),
        payload.get("first_pass_averages"),
    )
    story.append(Spacer(1, 8))
    story.append(bar_drawing)

    gain_rows = _format_iteration_gain_rows(payload)
    if gain_rows:
        story.append(Paragraph("Iterative Reasoning Gain", styles["SectionHeading"]))
        story.append(
            Paragraph(
                "What the iterative refinement loop added on top of the agent's 1st "
                "iteration. Best (Iter.) is the lowest-validity-score iteration seen — "
                "it equals the 1st iteration whenever that iteration was already the best one.",
                styles["Note"],
            )
        )
        story.append(Spacer(1, 4))
        story.append(
            _table(
                [
                    "Metric",
                    "Baseline",
                    "1st Iteration",
                    "Best (Iter.)",
                    "Gain",
                    "Direction",
                ],
                gain_rows,
                col_widths=[5 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm],
            )
        )
        gain_dist_rows = _format_iteration_gain_distribution_rows(sample_reports)
        if gain_dist_rows:
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    f"Per-Sample Gain Distribution (n={len(sample_reports)}, paired: best - first pass)",
                    styles["Note"],
                )
            )
            story.append(Spacer(1, 4))
            story.append(
                _table(
                    ["Metric", "Mean Gain", "Std Dev", "Min Gain", "Max Gain"],
                    gain_dist_rows,
                    col_widths=[6.5 * cm, 2.4 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm],
                )
            )

    if sample_reports:
        dist_rows = _format_distribution_rows(sample_reports)
        if dist_rows:
            story.append(
                Paragraph(
                    f"Distribution Summary (n={len(sample_reports)} samples)",
                    styles["SectionHeading"],
                )
            )
            story.append(
                Paragraph(
                    "The at-a-glance view for larger runs — read this before the per-sample appendix.",
                    styles["Note"],
                )
            )
            story.append(Spacer(1, 4))
            story.append(
                _table(
                    ["Metric", "Mean", "Std Dev", "Min", "Median", "Max"],
                    dist_rows,
                    col_widths=[6.5 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm],
                )
            )

        progression_drawing = _validity_progression_drawing(sample_reports)
        if progression_drawing is not None:
            story.append(Spacer(1, 10))
            story.append(progression_drawing)

        histogram_drawing = _validity_histogram_drawing(sample_reports)
        if histogram_drawing is not None:
            story.append(Spacer(1, 10))
            story.append(histogram_drawing)

    if sample_reports:
        story.append(PageBreak())
        story.append(
            Paragraph(
                f"Per-Sample Results (full appendix, n={len(sample_reports)})",
                styles["SectionHeading"],
            )
        )
        story.append(
            _table(
                [
                    "Sample",
                    "Exact",
                    "Best Score",
                    "Best Iter",
                    "Fragments",
                    "Pruned %",
                    "Confirmed Adj.",
                    "Duration",
                ],
                _format_per_sample_rows(sample_reports),
                col_widths=[1.6 * cm, 1.6 * cm, 2.5 * cm, 1.8 * cm, 2.2 * cm, 2.2 * cm, 2.8 * cm, 2.2 * cm],
                font_size=7,
            )
        )

    story.append(Spacer(1, 12))
    story.append(Paragraph("Quick Read", styles["SectionHeading"]))
    for line in (
        "Higher is better for all metrics except normalized edit distance.",
        "A positive delta means the reconstruction improved over the shuffled baseline.",
        "The validity score is pseudo-perplexity from the ESM-2 MLM; it measures plausibility, not exact-match correctness.",
        "Full per-iteration lever data (lever_values, changed_levers) is in samples.jsonl for auditability.",
    ):
        story.append(Paragraph(f"&bull; {line}", styles["Normal"]))

    doc.build(story)
    return pdf_path
