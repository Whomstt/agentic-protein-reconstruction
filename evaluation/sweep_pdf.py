from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from evaluation.pdf_report import _styles, _table
from evaluation.reporting import _format_duration
from evaluation.sweep_report import GAIN_HEADERS

GAIN_COL_WIDTHS = [5.5 * cm, 3.2 * cm, 3.6 * cm, 3.2 * cm, 3.2 * cm, 2.6 * cm]


def write_sweep_pdf(
    run_dir: Path,
    headers: list[str],
    rows: list[list[str]],
    succeeded: int,
    total: int,
    total_duration: float,
    gain_sections: list[tuple[str, list[list[str]]]] | None = None,
) -> Path:
    pdf_path = run_dir / "report.pdf"
    styles = _styles()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="Sweep Report",
    )

    story = [Paragraph("Sweep Report", styles["ReportTitle"])]

    story.append(Paragraph("Overview", styles["SectionHeading"]))
    for line in (
        f"Combinations run: {total}",
        f"Succeeded: {succeeded}/{total}",
        f"Total duration: {_format_duration(total_duration)}",
        f"Result folder: {run_dir.name}",
    ):
        story.append(Paragraph(f"&bull; {line}", styles["Normal"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Combo Comparison", styles["SectionHeading"]))
    story.append(
        _table(
            headers,
            rows,
            col_widths=[7 * cm, 2 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 5 * cm],
        )
    )

    if gain_sections:
        story.append(PageBreak())
        story.append(Paragraph("Iterative Reasoning Gain (per combo)", styles["SectionHeading"]))
        story.append(
            Paragraph(
                "What the iterative refinement loop added on top of the agent's 1st iteration, "
                "for every metric, in each combo. The best (iterative) candidate is always the "
                "one with the lowest validity score across all iterations, including the 1st.",
                styles["Normal"],
            )
        )
        for label, gain_rows in gain_sections:
            story.append(Spacer(1, 8))
            story.append(Paragraph(label, styles["Note"]))
            story.append(Spacer(1, 2))
            story.append(_table(GAIN_HEADERS, gain_rows, col_widths=GAIN_COL_WIDTHS))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Quick Read", styles["SectionHeading"]))
    for line in (
        "Each combo also has its own full report (charts, per-sample detail, PDF) under its Detail Folder in results/.",
        "This file is the cross-combo comparison; open a combo's own report.md/report.pdf for the full picture on that combo.",
    ):
        story.append(Paragraph(f"&bull; {line}", styles["Normal"]))

    doc.build(story)
    return pdf_path
