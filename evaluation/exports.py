"""CSV and LaTeX exports for the evaluation reports.

Markdown and LaTeX are rendered from the same row data (``Table``), so no number
is retyped by hand and the two cannot drift apart. stdlib only.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# LaTeX special characters that must be escaped in any cell we emit.
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # Text-mode < and > render as inverted punctuation in the default encoding.
    "<": r"$<$",
    ">": r"$>$",
}

# Unicode that shows up in labels but has no place in a .tex file.
_UNICODE_TO_LATEX = {
    "Δ": r"$\Delta$",
    "−": r"$-$",
    "±": r"$\pm$",
    "τ": r"$\tau$",
    "≈": r"$\approx$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "—": "---",
    "–": "--",
    "×": r"$\times$",
    "'": "'",
}


class Raw(str):
    """A data cell that is already LaTeX and must not be escaped.

    Data cells are escaped by default so a stray ``%`` or ``_`` in a value can
    never break the document. The one thing that legitimately needs markup in a
    cell is a species binomial, which has to be italic wherever it appears, so
    it is opted in explicitly rather than by relaxing the default.

    ``plain`` is what the same cell renders as outside LaTeX (CSV, markdown),
    where the markup would be noise.
    """

    plain: str

    def __new__(cls, latex: str, plain: str | None = None):
        cell = super().__new__(cls, latex)
        cell.plain = latex if plain is None else plain
        return cell


def latex_escape(text: str) -> str:
    if isinstance(text, Raw):
        return str(text)
    out = []
    for char in str(text):
        if char in _UNICODE_TO_LATEX:
            out.append(_UNICODE_TO_LATEX[char])
        elif char in _LATEX_ESCAPES:
            out.append(_LATEX_ESCAPES[char])
        else:
            out.append(char)
    return "".join(out)


def fmt(value, places: int = 4, signed: bool = False, dash: str = "n/a") -> str:
    """Format a number for a table cell. None/NaN render as ``dash``, never as 0."""
    if value is None:
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return dash
        if isinstance(value, int):
            return f"{value:+d}" if signed else f"{value:d}"
        return f"{value:+.{places}f}" if signed else f"{value:.{places}f}"
    return str(value)


def fmt_ci(interval, places: int = 4, dash: str = "n/a") -> str:
    """Render an Interval (or dict form) as 'point [low, high]'."""
    if interval is None:
        return dash
    data = interval if isinstance(interval, dict) else interval.as_dict()
    point, low, high = data.get("point"), data.get("low"), data.get("high")
    if point is None or (isinstance(point, float) and math.isnan(point)):
        return dash
    if low is None or (isinstance(low, float) and math.isnan(low)):
        return fmt(point, places)
    return f"{fmt(point, places)} [{fmt(low, places)}, {fmt(high, places)}]"


def fmt_p(value, dash: str = "n/a") -> str:
    """P-values: small ones as '<0.001' rather than a run of zeros."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return dash
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


# Use as a whole row to draw a rule between groups of rows.
MIDRULE = "\\midrule"


@dataclass
class Table:
    """One table, rendered to both markdown and booktabs LaTeX from one source.

    ``key`` names the .tex file; ``caption``/``label``/``notes`` are LaTeX-only.
    ``environment`` is ``"table"`` or ``"table*"``, the two-column-spanning float an
    IEEE template needs for a wide table. Prefer plain ``"table"`` where the columns
    fit: a single-column float can take the top or bottom of either column, so it
    lands beside the text that refers to it, whereas a spanning one only ever gets a
    page top and drifts pages away when several queue up. ``body_size``
    (e.g. ``"\\small"``) and ``col_sep`` (e.g. ``"4pt"``) buy the width that makes
    the difference, and apply to the tabular only — the caption keeps its own size.
    With ``raw_latex`` the caption, notes and headers are emitted verbatim instead
    of escaped, so a table written for the report can use maths and font commands;
    data cells are always escaped. ``provenance`` is filled in by
    :func:`stamp_tables`."""

    key: str
    headers: list[str]
    rows: list[list]
    caption: str = ""
    label: str = ""
    notes: str = ""
    column_spec: str = ""
    formatted: bool = True
    environment: str = "table"
    placement: str = "!tb"
    body_size: str = ""
    col_sep: str = ""
    raw_latex: bool = False
    provenance: dict = field(default_factory=dict)
    _extra: dict = field(default_factory=dict)

    @staticmethod
    def _md_cell(value) -> str:
        """A literal pipe would end the column early, so escape it."""
        if value is None:
            return ""
        if isinstance(value, Raw):
            value = value.plain
        return str(value).replace("|", r"\|")

    def to_markdown(self) -> str:
        header = "| " + " | ".join(self._md_cell(h) for h in self.headers) + " |"
        divider = "| " + " | ".join("---" for _ in self.headers) + " |"
        body = [
            "| " + " | ".join(self._md_cell(c) for c in row) + " |"
            for row in self.rows
            if row != MIDRULE
        ]
        return "\n".join([header, divider, *body])

    def _prose(self, text: str) -> str:
        return text if self.raw_latex else latex_escape(text)

    def _provenance_comment(self) -> list[str]:
        p = self.provenance
        lines = ["% Generated by evaluation/exports.py - do not edit by hand."]
        if p.get("source_run"):
            lines.append(f"% Source run:  {p['source_run']}")
        if p.get("source_file"):
            lines.append(f"% Source data: {p['source_file']}")
        if p.get("command"):
            lines.append(f"% Command:     {p['command']}")
        if p.get("n_rows") is not None:
            unit = p.get("row_unit", "samples")
            lines.append(f"% Rows:        {p['n_rows']} {unit}")
        if p.get("generated"):
            lines.append(f"% Generated:   {p['generated']}")
        if not p:
            lines.append("% Regenerate with: python -m evaluation.rebuild --all")
        return lines

    def to_latex(self, comments: bool = True) -> str:
        spec = self.column_spec or ("l" + "r" * (len(self.headers) - 1))
        env = self.environment or "table"
        lines = (self._provenance_comment() if comments else []) + [
            rf"\begin{{{env}}}[{self.placement}]",
            r"  \centering",
        ]
        if self.caption:
            lines.append(rf"  \caption{{{self._prose(self.caption)}}}")
        if self.label:
            lines.append(rf"  \label{{{self.label}}}")
        # After the caption so the caption keeps the class's own size; inside the
        # float, so both are local to it.
        if self.body_size:
            lines.append(f"  {self.body_size}")
        if self.col_sep:
            lines.append(rf"  \setlength{{\tabcolsep}}{{{self.col_sep}}}")
        lines += [
            rf"  \begin{{tabular}}{{{spec}}}",
            r"    \toprule",
            "    " + " & ".join(self._prose(h) for h in self.headers) + r" \\",
            r"    \midrule",
        ]
        for row in self.rows:
            if row == MIDRULE:
                lines.append("    " + MIDRULE)
                continue
            cells = ["" if c is None else latex_escape(c) for c in row]
            lines.append("    " + " & ".join(cells) + r" \\")
        lines += [r"    \bottomrule", r"  \end{tabular}"]
        if self.notes:
            lines.append(rf"  \par\smallskip\footnotesize{{{self._prose(self.notes)}}}")
        lines += [rf"\end{{{env}}}", ""]
        return "\n".join(lines)


def stamp_tables(
    tables: list[Table],
    source_run: str,
    command: str,
    n_rows: int,
    source_file: str = "samples.jsonl",
    row_unit: str = "samples",
    generated: str | None = None,
) -> list[Table]:
    """Fill in each table's provenance header in place.

    A table that already set a key (typically ``n_rows``, when its numbers rest
    on a subset of the samples) keeps its own value; only missing keys are
    filled, from one timestamp shared by the whole pass.
    """
    stamp = generated or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    defaults = {
        "source_run": source_run,
        "source_file": source_file,
        "command": command,
        "n_rows": n_rows,
        "row_unit": row_unit,
        "generated": stamp,
    }
    for table in tables:
        for key, value in defaults.items():
            table.provenance.setdefault(key, value)
    return tables


def write_tables_tex(
    tables: list[Table],
    out_dir: Path,
    comments: bool = True,
    extra_inputs: list[str] | None = None,
    merge_index: bool = False,
) -> list[Path]:
    """One .tex file per table, plus an ``all_tables.tex`` that inputs them in
    order. ``extra_inputs`` names files written elsewhere to append to the index.

    ``merge_index`` keeps the \\input lines already in ``all_tables.tex`` whose
    .tex file is still present and appends the new ones, so regenerating one
    table does not drop the others from the index and a deleted table does not
    linger in it. Without it the index is rewritten from this call's tables alone.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for table in tables:
        path = out_dir / f"{table.key}.tex"
        path.write_text(table.to_latex(comments=comments), encoding="utf-8")
        written.append(path)
    stems = [p.stem for p in written] + list(extra_inputs or [])

    index = out_dir / "all_tables.tex"
    if merge_index and index.exists():
        existing = [
            stem
            for stem in re.findall(r"\\input\{([^}]*)\}", index.read_text(encoding="utf-8"))
            if (out_dir / f"{stem}.tex").exists()
        ]
        stems = existing + [s for s in stems if s not in existing]

    header = "% Generated by evaluation/exports.py - do not edit by hand.\n" if comments else ""
    index.write_text(
        header + "\n".join(rf"\input{{{stem}}}" for stem in stems) + "\n",
        encoding="utf-8",
    )
    return written


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def _stringify(value):
    """None -> empty cell; NaN -> empty cell. An empty cell is unambiguous in a
    CSV; 'nan' silently becomes a string in most readers and 0 would be a lie."""
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return value


def write_rows_csv(rows: list[dict], path: Path, columns: list[str] | None = None) -> Path:
    """Write dict rows to CSV with a stable, union-of-keys header.

    Column order is: the order keys first appear across rows. Runs written
    before a field existed simply leave that cell empty, so old and new runs
    concatenate into one cross-run file without error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify(row.get(k)) for k in columns})
    return path


def write_summary_csv(records: list[dict], path: Path) -> Path:
    """Aggregate metrics with CIs, one row per (arm, metric)."""
    return write_rows_csv(records, path)
