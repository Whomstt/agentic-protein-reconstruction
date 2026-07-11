"""Builds one combined report across every combo in a sweep, instead of
leaving results scattered across N separate per-combo folders. Each combo
keeps its own full report (charts, per-sample detail); this adds a single
cross-combo comparison on top, in its own results/ folder.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evaluation.reporting import (
    RESULTS_ROOT,
    _format_duration,
    _format_iteration_gain_rows,
    _markdown_table,
    _sanitize,
    first_pass_label,
    selected_best_label,
)


def _load_combo_payload(result_dir: str | None) -> dict | None:
    if not result_dir:
        return None
    summary_path = Path(result_dir) / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open(encoding="utf-8") as f:
        return json.load(f)


def _combo_row(entry: dict) -> list[str]:
    combo = entry["combo"]
    label = ", ".join(f"{k}={v}" for k, v in combo.items())
    duration_text = _format_duration(entry.get("duration_seconds"))
    detail_folder = Path(entry["result_dir"]).name if entry.get("result_dir") else "-"

    if entry["status"] != "ok":
        return [label, "FAILED", "-", "-", "-", duration_text, entry.get("error") or detail_folder]

    payload = _load_combo_payload(entry.get("result_dir"))
    if payload is None:
        return [label, "ok (no summary.json found)", "-", "-", "-", duration_text, detail_folder]

    recon = payload.get("recon_averages", {})
    n = payload.get("sample_count", 0)
    return [
        label,
        str(n),
        f"{recon.get('exact_match', 0):.4f}",
        f"{recon.get('similarity', 0):.4f}",
        f"{recon.get('kendall_tau', 0):.4f}",
        duration_text,
        detail_folder,
    ]


def _combo_gain_table(entry: dict) -> tuple[str, list[list[str]]] | None:
    """Full baseline/1st-iteration/best per-metric uplift table for one combo, i.e.
    what the iterative refinement loop added on top of the agent's 1st iteration —
    the same table each combo's own report.md shows, pulled into the combined
    sweep report so it's readable without opening every combo folder."""
    if entry["status"] != "ok":
        return None
    payload = _load_combo_payload(entry.get("result_dir"))
    if payload is None:
        return None
    gain_rows = _format_iteration_gain_rows(payload)
    if not gain_rows:
        return None
    label = ", ".join(f"{k}={v}" for k, v in entry["combo"].items())
    return label, gain_rows


def _first_combo_config(manifest: list[dict]) -> dict:
    """The run config from the first successful combo. All combos in a sweep
    share run.method/calling_mode/iteration1_deterministic (only organism,
    replica_count and mlm_profile are swept), so this is enough to derive the
    deterministic-vs-agentic column labels for the whole cross-combo report."""
    for entry in manifest:
        payload = _load_combo_payload(entry.get("result_dir"))
        if payload is not None:
            return payload.get("config", {})
    return {}


def write_sweep_report(manifest: list[dict], sweep_cfg: dict, total_duration: float) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    run_dir = RESULTS_ROOT / f"{timestamp}_{_sanitize('sweep')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    combo_config = _first_combo_config(manifest)
    is_sequential = combo_config.get("run", {}).get("method") == "sequential"
    fp_label = first_pass_label(combo_config)
    best_label = selected_best_label(combo_config)
    best_short = "Reconstructed" if is_sequential else "Agentic Best"
    gain_headers = [
        "Metric",
        "Shuffled Baseline",
        fp_label,
        best_label,
        "Gain vs. 1st Pass",
        "Direction",
    ]

    succeeded = sum(1 for m in manifest if m["status"] == "ok")
    rows = [_combo_row(entry) for entry in manifest]
    headers = [
        "Combo",
        "Samples",
        "Exact Match",
        "Similarity",
        "Kendall Tau",
        "Duration",
        "Detail Folder",
    ]

    gain_sections = [_combo_gain_table(entry) for entry in manifest]
    gain_sections = [section for section in gain_sections if section is not None]

    lines = [
        "# Sweep Report",
        "",
        "## Overview",
        f"- Combinations run: {len(manifest)}",
        f"- Succeeded: {succeeded}/{len(manifest)}",
        f"- Total duration: {_format_duration(total_duration)}",
        f"- Result folder: `{run_dir.name}`",
        "",
        "## Combo Comparison",
        f"Metric columns report the **{best_label}** per combo.",
        "",
        _markdown_table(headers, rows),
        "",
    ]

    if gain_sections:
        lines.append("## Iterative Reasoning Gain (per combo)")
        lines.append(
            f"What the iterative refinement loop added on top of the {fp_label}, for every "
            f"metric, in each combo. **{best_short}** is always the candidate with the lowest "
            "validity score across all iterations (including the 1st)."
        )
        lines.append("")
        for label, gain_rows in gain_sections:
            lines.append(f"### {label}")
            lines.append(_markdown_table(gain_headers, gain_rows))
            lines.append("")

    lines.extend(
        [
            "## Quick Read",
            "- Each combo also has its own full `report.md` (charts, per-sample detail) under `Detail Folder` in `results/`.",
            "- This file is the cross-combo comparison; open a combo's own `report.md` for the full picture on that combo.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_payload = {
        "manifest": manifest,
        "sweep_config": sweep_cfg,
        "total_duration_seconds": total_duration,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return run_dir
