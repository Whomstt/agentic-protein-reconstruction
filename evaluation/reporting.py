from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from evaluation.metrics import (
    LOWER_IS_BETTER,
    METRIC_NAMES,
    print_comparison,
    rank_concordance,
)

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

# Above this many samples, the validity-progression chart switches from
# one line per sample to a mean +/- range band — per-sample lines and a
# same-sized legend stop being readable well before n=100.
AGGREGATE_LINE_THRESHOLD = 12

METRIC_RANGES = {
    "exact_match": (0.0, 1.0),
    "similarity": (0.0, 1.0),
    "adjacent_pair_acc": (0.0, 1.0),
    "longest_correct_run": (0.0, 1.0),
    "kendall_tau": (-1.0, 1.0),
}

BASELINE_COLOR = "#9aa5b1"
FIRST_PASS_COLOR = "#f59e0b"
RECON_COLOR = "#3b82f6"
LINE_PALETTE = ["#3b82f6", "#f97316", "#10b981", "#e11d48", "#8b5cf6", "#eab308"]


def _percentile(sorted_values: list[float], pct: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * pct
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def distribution_stats(values) -> dict | None:
    """Summary stats (mean/std/quartiles) for a list of numeric values, ignoring
    None/NaN/inf entries. Returns None if nothing usable is left."""
    vals = sorted(
        v
        for v in values
        if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)
    )
    if not vals:
        return None
    n = len(vals)
    return {
        "n": n,
        "mean": sum(vals) / n,
        "std": statistics.pstdev(vals) if n > 1 else 0.0,
        "min": vals[0],
        "p25": _percentile(vals, 0.25),
        "median": _percentile(vals, 0.5),
        "p75": _percentile(vals, 0.75),
        "max": vals[-1],
    }


def _validity_best_so_far_series(sample_reports: list[dict]) -> list[dict]:
    """Per-sample list of {"label", "values"} where values[i] is the best
    validity score seen through iteration i+1 (running minimum)."""
    series = []
    for report in sample_reports:
        history = report.get("iteration_history", [])
        values = []
        best_so_far = None
        for record in history:
            score = record.get("validity_score")
            if isinstance(score, (int, float)) and not math.isinf(score):
                best_so_far = score if best_so_far is None else min(best_so_far, score)
            values.append(best_so_far)
        if any(v is not None for v in values):
            series.append({"label": f"Sample {report['index']}", "values": values})
    return series


def _aggregate_iteration_stats(series: list[dict]) -> tuple[int, list, list, list]:
    """Per-iteration mean/min/max of validity scores across samples. Returns
    (max_len, mean_line, min_line, max_line); entries are None where no sample
    had data yet at that iteration."""
    max_len = max((len(s["values"]) for s in series), default=0)
    mean_line, min_line, max_line = [], [], []
    for idx in range(max_len):
        vals = [
            s["values"][idx]
            for s in series
            if idx < len(s["values"]) and s["values"][idx] is not None
        ]
        if vals:
            mean_line.append(sum(vals) / len(vals))
            min_line.append(min(vals))
            max_line.append(max(vals))
        else:
            mean_line.append(None)
            min_line.append(None)
            max_line.append(None)
    return max_len, mean_line, min_line, max_line


def validity_score_histogram_bins(sample_reports: list[dict], bins: int = 10):
    """Returns (lo, hi, bin_width, counts) for best_validity_score across samples,
    or None if there's nothing to bin."""
    vals = [
        r.get("best_validity_score")
        for r in sample_reports
        if isinstance(r.get("best_validity_score"), (int, float))
        and not math.isinf(r.get("best_validity_score"))
    ]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if lo == hi:
        hi = lo + 1e-6
    bin_width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        idx = min(bins - 1, int((v - lo) / bin_width))
        counts[idx] += 1
    return lo, hi, bin_width, counts


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _iteration1_is_deterministic(config: dict) -> bool:
    """Whether iteration 1 (the 'first pass') runs with no LLM call. This is true
    only in single_call mode with run.iteration1_deterministic set: react mode
    always drives iteration 1 through the LLM regardless of that flag, so the flag
    alone can't be trusted to describe what actually happened."""
    run = config.get("run", {})
    if run.get("calling_mode") == "react":
        return False
    return run.get("iteration1_deterministic", True)


def first_pass_label(config: dict) -> str:
    """Label for the iteration-1 ('first pass') column. When iteration 1 is a
    no-LLM config-defaults pass, it is labelled as deterministic so the number is
    never mistaken for an agent decision; otherwise it is a genuine LLM lever
    choice (see CLAUDE.md's research-validity note)."""
    if _iteration1_is_deterministic(config):
        return "Deterministic (config defaults, no LLM)"
    return "Agentic 1st Pass (LLM)"


def selected_best_label(config: dict) -> str:
    """Label for the final chosen reconstruction column. For agentic runs this is
    the iteratively-selected best-validity candidate; for sequential runs it is
    just the deterministic pipeline output."""
    if config.get("run", {}).get("method") == "sequential":
        return "Reconstructed (deterministic pipeline)"
    return "Agentic Best (iteratively selected)"


def run_type_summary(
    config: dict, has_control: bool = False, has_oracle: bool = False
) -> list[str]:
    """Markdown lines spelling out, for THIS run's config, exactly what each
    compared column means — so a reader never has to guess which numbers came from
    the deterministic pipeline and which are the LLM agent's iteratively-selected
    result."""
    run = config.get("run", {})
    if run.get("method") == "sequential":
        return [
            "## How to Read This Report",
            "**Run type: Sequential (deterministic, no LLM).** Each sample gets a "
            "single fixed reconstruction from the tool pipeline — there is no agent, "
            "no iteration, and no LLM call. **Shuffled Baseline** is a random fragment "
            "ordering (the floor); **Reconstructed** is the deterministic pipeline's output.",
            "",
        ]

    calling_mode = run.get("calling_mode", "react")
    lines = [
        "## How to Read This Report",
        f"**Run type: Agentic ({calling_mode}).** Each metric compares:",
        "",
        "- **Shuffled Baseline** — a random fragment ordering. The floor, not a method.",
    ]
    if _iteration1_is_deterministic(config):
        lines.append(
            "- **Deterministic (config defaults)** — the non-agentic baseline: iteration 1 run "
            "with the fixed `search.default_levers` and **no LLM call**. The agent refines from it."
        )
    else:
        lines.append(
            "- **Agentic 1st Pass** — iteration 1 is itself an LLM lever decision "
            "(`run.iteration1_deterministic=false`); there is no deterministic baseline inside this run."
        )
    if has_control:
        lines.append(
            "- **Control (non-LLM)** — the matched-budget control arm: the SAME iteration "
            "budget, SAME fixed tool pipeline and SAME best-validity selection as the agentic "
            "arm, but the five levers are chosen by a non-LLM policy (random/grid) instead of the "
            "LLM, and it runs paired on the same protein with 0 LLM calls. **`Δ Agentic − Control` "
            'is the isolated value of the LLM\'s reasoning** — it separates "the agent reasons '
            'well" from "trying several candidates and keeping the best-validity one helps."'
        )
    lines.append(
        "- **Agentic Best** — the agent's result: iterations 2+ are LLM lever choices, and the "
        "kept candidate is the best-validity one across all iterations (subject to "
        "`search.improvement_margin`). Since iteration 1 (the deterministic baseline) is in the "
        'candidate set, read the **true-metric** columns for the real "does the agent help?" answer.'
    )
    if has_oracle:
        lines.append(
            "- **Oracle (ceiling)** — for each metric, the best value achievable by selecting "
            "among the candidates the agent actually generated. Not a method (it peeks at the "
            "ground truth); the **Oracle − Agentic** gap is what the imperfect (~57–61%) validity "
            "concordance leaves on the table — reachable by better selection alone, no new candidate."
        )
    lines.append("")
    return lines


def render_metric_bar_chart_svg(
    baseline_averages: dict,
    recon_averages: dict,
    first_pass_averages: dict | None = None,
    first_pass_label_text: str = "1st Pass",
    recon_label_text: str = "Agentic Best",
) -> str:
    """Grouped horizontal bar chart comparing baseline vs. (optionally first-pass
    vs.) reconstructed metrics. When first_pass_averages is given, a third bar
    isolates what iterative refinement added on top of a single-shot pass."""
    row_h = 34 if first_pass_averages is None else 44
    label_w = 190
    chart_w = 320
    left_pad = 12
    top_pad = 44
    keys = list(METRIC_NAMES.keys())
    height = top_pad + row_h * len(keys) + 40
    width = left_pad + label_w + chart_w + 70

    bars = []
    for i, key in enumerate(keys):
        lo, hi = METRIC_RANGES[key]
        base_val = baseline_averages.get(key, 0.0)
        recon_val = recon_averages.get(key, 0.0)
        y = top_pad + i * row_h

        def bar_len(v):
            v = max(lo, min(hi, v))
            return (v - lo) / (hi - lo) * chart_w

        base_x = left_pad + label_w
        bars.append(
            f'<text x="{left_pad}" y="{y + 20}" font-size="12" '
            f'fill="currentColor">{_svg_escape(METRIC_NAMES[key])}</text>'
        )
        bars.append(
            f'<rect x="{base_x}" y="{y + 2}" width="{bar_len(base_val):.1f}" height="10" '
            f'fill="{BASELINE_COLOR}" rx="2"/>'
        )
        bars.append(
            f'<text x="{base_x + bar_len(base_val) + 6:.1f}" y="{y + 11}" font-size="10" '
            f'fill="{BASELINE_COLOR}">{base_val:.3f}</text>'
        )

        next_row_y = y + 16
        if first_pass_averages is not None:
            first_pass_val = first_pass_averages.get(key, 0.0)
            bars.append(
                f'<rect x="{base_x}" y="{next_row_y}" width="{bar_len(first_pass_val):.1f}" height="10" '
                f'fill="{FIRST_PASS_COLOR}" rx="2"/>'
            )
            bars.append(
                f'<text x="{base_x + bar_len(first_pass_val) + 6:.1f}" y="{next_row_y + 9}" font-size="10" '
                f'fill="{FIRST_PASS_COLOR}">{first_pass_val:.3f}</text>'
            )
            next_row_y += 14

        bars.append(
            f'<rect x="{base_x}" y="{next_row_y}" width="{bar_len(recon_val):.1f}" height="10" '
            f'fill="{RECON_COLOR}" rx="2"/>'
        )
        bars.append(
            f'<text x="{base_x + bar_len(recon_val) + 6:.1f}" y="{next_row_y + 9}" font-size="10" '
            f'fill="{RECON_COLOR}">{recon_val:.3f}</text>'
        )

    legend_y = top_pad + row_h * len(keys) + 20
    legend_entries = [(BASELINE_COLOR, "Shuffled baseline")]
    if first_pass_averages is not None:
        legend_entries.append((FIRST_PASS_COLOR, first_pass_label_text))
    legend_entries.append((RECON_COLOR, recon_label_text))
    legend_parts = []
    for i, (color, label) in enumerate(legend_entries):
        lx = left_pad + label_w + i * 155
        legend_parts.append(
            f'<rect x="{lx}" y="{legend_y - 9}" width="10" height="10" fill="{color}" rx="2"/>'
            f'<text x="{lx + 16}" y="{legend_y}" font-size="11" fill="currentColor">{label}</text>'
        )
    legend = "".join(legend_parts)

    title_text = (
        f"Metric Comparison: Shuffled Baseline vs. {first_pass_label_text} vs. {recon_label_text}"
        if first_pass_averages is not None
        else f"Metric Comparison: Baseline vs. {recon_label_text}"
    )
    title = f'<text x="12" y="20" font-size="14" font-weight="600" fill="currentColor">{title_text}</text>'

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="sans-serif" color="#1f2937">'
        f"{title}{''.join(bars)}{legend}</svg>"
    )


def render_validity_progression_svg(sample_reports: list[dict]) -> str:
    """Line chart of best-so-far validity score across iterations.

    With few samples, draws one labeled line per sample (small-multiples view).
    Beyond AGGREGATE_LINE_THRESHOLD samples, per-sample lines and a same-sized
    legend stop being readable, so it switches to a mean line with a min-max
    band across samples instead.
    """
    series = _validity_best_so_far_series(sample_reports)
    if not series:
        return ""

    aggregate = len(series) > AGGREGATE_LINE_THRESHOLD

    width, height = 620, 300
    pad_l, pad_r, pad_t, pad_b = 56, 20, 36, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_vals = [v for s in series for v in s["values"] if v is not None]
    if not all_vals:
        return ""
    y_min, y_max = min(all_vals), max(all_vals)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    max_len = max((len(s["values"]) for s in series), default=1)
    max_len = max(max_len, 2)

    def px(idx):
        return pad_l + (idx / (max_len - 1)) * plot_w

    def py(val):
        return pad_t + (1 - (val - y_min) / (y_max - y_min)) * plot_h

    gridlines = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + frac * plot_h
        val = y_max - frac * (y_max - y_min)
        gridlines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="currentColor" stroke-opacity="0.12"/>'
        )
        gridlines.append(
            f'<text x="{pad_l - 8}" y="{y + 3:.1f}" font-size="10" text-anchor="end" '
            f'fill="currentColor">{val:.2f}</text>'
        )

    x_ticks = []
    for idx in range(max_len):
        x_ticks.append(
            f'<text x="{px(idx):.1f}" y="{height - pad_b + 16}" font-size="10" '
            f'text-anchor="middle" fill="currentColor">{idx + 1}</text>'
        )

    lines = []
    if aggregate:
        _, mean_line, min_line, max_line = _aggregate_iteration_stats(series)
        band_points = [(idx, v) for idx, v in enumerate(min_line) if v is not None]
        band_points_top = [
            (idx, v) for idx, v in enumerate(max_line) if v is not None
        ][::-1]
        if band_points and band_points_top:
            band_path = " ".join(
                f"{'M' if j == 0 else 'L'}{px(idx):.1f},{py(v):.1f}"
                for j, (idx, v) in enumerate(band_points + band_points_top)
            )
            lines.append(
                f'<path d="{band_path} Z" fill="{RECON_COLOR}" fill-opacity="0.12" stroke="none"/>'
            )
        mean_points = [(idx, v) for idx, v in enumerate(mean_line) if v is not None]
        if mean_points:
            mean_path = " ".join(
                f"{'M' if j == 0 else 'L'}{px(idx):.1f},{py(v):.1f}"
                for j, (idx, v) in enumerate(mean_points)
            )
            lines.append(
                f'<path d="{mean_path}" fill="none" stroke="{RECON_COLOR}" stroke-width="2.5"/>'
            )
            for idx, v in mean_points:
                lines.append(
                    f'<circle cx="{px(idx):.1f}" cy="{py(v):.1f}" r="3" fill="{RECON_COLOR}"/>'
                )
        lines.append(
            f'<text x="{pad_l + 6}" y="{pad_t + 12}" font-size="10" '
            f'fill="{RECON_COLOR}">Mean across {len(series)} samples (band = min-max)</text>'
        )
    else:
        for i, s in enumerate(series):
            color = LINE_PALETTE[i % len(LINE_PALETTE)]
            points = [
                (idx, v) for idx, v in enumerate(s["values"]) if v is not None
            ]
            if not points:
                continue
            path = " ".join(
                f"{'M' if j == 0 else 'L'}{px(idx):.1f},{py(v):.1f}"
                for j, (idx, v) in enumerate(points)
            )
            lines.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for idx, v in points:
                lines.append(
                    f'<circle cx="{px(idx):.1f}" cy="{py(v):.1f}" r="2.5" fill="{color}"/>'
                )
            lines.append(
                f'<text x="{pad_l + 6}" y="{pad_t + 12 + i * 13}" font-size="10" '
                f'fill="{color}">{_svg_escape(s["label"])}</text>'
            )

    title = (
        '<text x="12" y="20" font-size="14" font-weight="600" fill="currentColor">'
        "Best Validity Score by Iteration (lower = better)</text>"
    )
    x_label = (
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 4}" font-size="10" '
        f'text-anchor="middle" fill="currentColor">Iteration</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="sans-serif" color="#1f2937">'
        f"{title}{''.join(gridlines)}{''.join(x_ticks)}{''.join(lines)}{x_label}</svg>"
    )


def render_validity_histogram_svg(sample_reports: list[dict], bins: int = 10) -> str:
    """Histogram of each sample's best validity score — the key "does this
    setup work reliably" view once n grows large enough that per-sample
    inspection stops being practical."""
    binned = validity_score_histogram_bins(sample_reports, bins=bins)
    if binned is None:
        return ""
    lo, hi, bin_width, counts = binned

    width, height = 520, 260
    pad_l, pad_r, pad_t, pad_b = 46, 16, 36, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_count = max(counts) or 1
    bar_w = plot_w / len(counts)

    bars = []
    for i, count in enumerate(counts):
        bar_h = (count / max_count) * plot_h
        x = pad_l + i * bar_w
        y = pad_t + (plot_h - bar_h)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 2, 0):.1f}" '
            f'height="{bar_h:.1f}" fill="{RECON_COLOR}" rx="2"/>'
        )
        if count:
            bars.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" font-size="9" '
                f'text-anchor="middle" fill="currentColor">{count}</text>'
            )

    axis = [
        f'<text x="{pad_l}" y="{height - pad_b + 16}" font-size="10" '
        f'fill="currentColor">{lo:.2f}</text>',
        f'<text x="{pad_l + plot_w:.1f}" y="{height - pad_b + 16}" font-size="10" '
        f'text-anchor="end" fill="currentColor">{hi:.2f}</text>',
    ]
    total = sum(counts)
    title = (
        '<text x="12" y="20" font-size="14" font-weight="600" fill="currentColor">'
        f"Best Validity Score Distribution (n={total})</text>"
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="sans-serif" color="#1f2937">'
        f"{title}{''.join(bars)}{''.join(axis)}</svg>"
    )


def build_config_snapshot(cfg: dict) -> dict:
    return {
        "misc": {
            "seed": cfg["misc"].get("seed"),
            "device": cfg["misc"].get("device"),
        },
        "run": {
            "method": cfg.get("run", {}).get("method"),
            "calling_mode": cfg.get("run", {}).get("calling_mode"),
            "iteration1_deterministic": cfg.get("run", {}).get(
                "iteration1_deterministic", True
            ),
        },
        "mlm_model": {
            "profile": cfg["mlm_model"].get("profile"),
            "type": cfg["mlm_model"].get("type"),
            "name": cfg["mlm_model"].get("name"),
            "batch_size": cfg["mlm_model"].get("batch_size"),
            "max_length": cfg["mlm_model"].get("max_length"),
        },
        "validity_model": {
            "name": cfg.get("validity_model", {}).get("name"),
            "batch_size": cfg.get("validity_model", {}).get("batch_size"),
            "max_length": cfg.get("validity_model", {}).get("max_length"),
        },
        "llm_model": {
            # Resolved llm profiles carry the deployment under "model"; fall back
            # to "name" for any profile that used that key instead.
            "name": cfg["llm_model"].get("model") or cfg["llm_model"].get("name"),
            "sampling": dict(cfg["llm_model"].get("sampling") or {}),
        },
        "search": {
            "max_iterations": cfg.get("search", {}).get("max_iterations"),
            "early_stop_patience": cfg.get("search", {}).get("early_stop_patience"),
            "default_beam_width": cfg.get("search", {})
            .get("default_levers", {})
            .get("beam_width"),
            "default_junction_window": cfg.get("search", {})
            .get("default_levers", {})
            .get("junction_window"),
        },
        "data": {
            "organism": cfg["data"].get("organism"),
            "organism_display_name": cfg["data"].get("organism_display_name"),
            "test_samples": cfg["data"].get("test_samples"),
            "replica_count": cfg["data"].get(
                "replica_count", cfg["data"].get("sample_count")
            ),
            "missed_cleavage_ratio": cfg["data"].get("missed_cleavage_ratio"),
            "active_fragmented_split": cfg["data"].get("active_fragmented_split"),
        },
    }


def _format_metric_block(metrics: dict[str, float]) -> str:
    rows = []
    for key, label in METRIC_NAMES.items():
        rows.append(f"{label}: {metrics[key]:.4f}")
    return "\n".join(rows)


def _format_optional_float(value) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def _format_config_rows(config: dict) -> list[list[str]]:
    run = config.get("run", {})
    method = run.get("method")
    method_row = [["Run Method", str(method)]]
    if method != "sequential":
        method_row.append(["Calling Mode", str(run.get("calling_mode"))])
    return [
        *method_row,
        ["Device", str(config["misc"]["device"])],
        ["Seed", str(config["misc"]["seed"])],
        [
            "Dataset",
            str(
                config["data"].get("organism_display_name")
                or config["data"].get("organism")
            ),
        ],
        ["Test Samples", str(config["data"]["test_samples"])],
        ["Replica Count", str(config["data"]["replica_count"])],
        ["Missed Cleavage Ratio", str(config["data"]["missed_cleavage_ratio"])],
        ["MLM Model", str(config["mlm_model"]["name"])],
        ["MLM Type", str(config["mlm_model"]["type"])],
        ["MLM Batch Size", str(config["mlm_model"]["batch_size"])],
        ["MLM Max Length", str(config["mlm_model"]["max_length"])],
        ["Beam Width", str(config["search"]["default_beam_width"])],
        ["Junction Window", str(config["search"]["default_junction_window"])],
        ["Validity Model", str(config.get("validity_model", {}).get("name"))],
        ["Max Iterations", str(config.get("search", {}).get("max_iterations"))],
        [
            "Iteration 1 Mode",
            "Deterministic — config defaults (no LLM call)"
            if _iteration1_is_deterministic(config)
            else "LLM (genuine agent decision)",
        ],
        [
            "Early Stop Patience",
            str(config.get("search", {}).get("early_stop_patience")),
        ],
        ["LLM Model", str(config["llm_model"]["name"])],
        ["LLM Sampling", _format_sampling(config["llm_model"].get("sampling"))],
    ]


def _format_sampling(sampling: dict | None) -> str:
    """Render only the sampling knobs that are actually set, so the report shows
    what was sent to the model (reasoning models omit temperature/top_p)."""
    if not sampling:
        return "model defaults"
    parts = [f"{key}={value}" for key, value in sampling.items() if value is not None]
    return ", ".join(parts) if parts else "model defaults"


def _format_metric_rows(payload: dict) -> list[list[str]]:
    rows = []
    for key, label in METRIC_NAMES.items():
        baseline = payload["baseline_averages"][key]
        recon = payload["recon_averages"][key]
        delta = payload["delta"][key]
        lower_better = key in LOWER_IS_BETTER
        better = "down" if lower_better else "up"
        if lower_better:
            status = "better" if delta < 0 else ("worse" if delta > 0 else "same")
        else:
            status = "better" if delta > 0 else ("worse" if delta < 0 else "same")
        rows.append(
            [
                label,
                f"{baseline:.4f}",
                f"{recon:.4f}",
                f"{delta:+.4f}",
                status,
                better,
            ]
        )
    return rows


def _format_iteration_gain_rows(payload: dict) -> list[list[str]]:
    """Baseline vs. first-pass (iteration 1) vs. best (iterative) per metric —
    isolates what the iterative refinement loop added on top of a single shot."""
    baseline_averages = payload.get("baseline_averages", {})
    first_pass_averages = payload.get("first_pass_averages")
    recon_averages = payload.get("recon_averages", {})
    if not first_pass_averages:
        return []
    rows = []
    for key, label in METRIC_NAMES.items():
        baseline = baseline_averages.get(key, 0.0)
        first_pass = first_pass_averages.get(key, 0.0)
        best = recon_averages.get(key, 0.0)
        gain = best - first_pass
        if key in LOWER_IS_BETTER:
            status = "better" if gain < 0 else ("worse" if gain > 0 else "same")
        else:
            status = "better" if gain > 0 else ("worse" if gain < 0 else "same")
        rows.append(
            [
                label,
                f"{baseline:.4f}",
                f"{first_pass:.4f}",
                f"{best:.4f}",
                f"{gain:+.4f}",
                status,
            ]
        )
    return rows


def _format_iteration_gain_distribution_rows(sample_reports: list[dict]) -> list[list[str]]:
    """Paired per-sample gain (best - first pass) mean/std across samples —
    shows how reliably iterative refinement helps, not just the average shift."""
    rows = []
    for key, label in METRIC_NAMES.items():
        stats = distribution_stats(
            [r.get("iteration_gain", {}).get(key) for r in sample_reports]
        )
        if stats is None:
            continue
        rows.append(
            [
                label,
                f"{stats['mean']:+.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:+.4f}",
                f"{stats['max']:+.4f}",
            ]
        )
    validity_gain_stats = distribution_stats(
        [
            (r.get("best_validity_score") or 0) - (r.get("first_pass_validity_score") or 0)
            for r in sample_reports
            if isinstance(r.get("best_validity_score"), (int, float))
            and isinstance(r.get("first_pass_validity_score"), (int, float))
        ]
    )
    if validity_gain_stats is not None:
        rows.append(
            [
                "Best Validity Score (junction+overlap blend, lower=better)",
                f"{validity_gain_stats['mean']:+.4f}",
                f"{validity_gain_stats['std']:.4f}",
                f"{validity_gain_stats['min']:+.4f}",
                f"{validity_gain_stats['max']:+.4f}",
            ]
        )
    return rows


def _control_label(payload: dict) -> str:
    policy = payload.get("control_policy") or "random"
    return f"Control ({policy}, no LLM)"


def _format_multi_arm_benchmark(
    payload: dict, fp_label: str, best_label: str
) -> tuple[list[str] | None, list[list[str]] | None]:
    """Headline benchmark row set, widened to include the matched-budget control
    arm and the oracle ceiling when those were produced. Columns are emitted only
    for arms that exist, so a plain run (no control/oracle) renders exactly the
    original three-arm table. Returns (headers, rows) or (None, None) when there
    is no first-pass arm to compare against (sequential runs)."""
    baseline = payload.get("baseline_averages", {})
    first_pass = payload.get("first_pass_averages")
    recon = payload.get("recon_averages", {})
    control = payload.get("control_averages")
    oracle = payload.get("oracle_averages")
    if not first_pass:
        return None, None

    headers = ["Metric", "Shuffled Baseline", fp_label]
    if control:
        headers.append(_control_label(payload))
    headers.append(best_label)
    if oracle:
        headers.append("Oracle (ceiling)")
    headers.append("Δ Agentic − Deterministic")
    if control:
        headers.append("Δ Agentic − Control")
    headers.append("Direction")

    rows = []
    for key, label in METRIC_NAMES.items():
        rc = recon.get(key, 0.0)
        fp = first_pass.get(key, 0.0)
        row = [label, f"{baseline.get(key, 0.0):.4f}", f"{fp:.4f}"]
        if control:
            row.append(f"{control.get(key, 0.0):.4f}")
        row.append(f"{rc:.4f}")
        if oracle:
            row.append(f"{oracle.get(key, 0.0):.4f}")
        det_delta = rc - fp
        row.append(f"{det_delta:+.4f}")
        if control:
            row.append(f"{rc - control.get(key, 0.0):+.4f}")
        if key in LOWER_IS_BETTER:
            status = "better" if det_delta < 0 else ("worse" if det_delta > 0 else "same")
        else:
            status = "better" if det_delta > 0 else ("worse" if det_delta < 0 else "same")
        row.append(status)
        rows.append(row)
    return headers, rows


def _format_agent_vs_control_distribution_rows(
    sample_reports: list[dict],
) -> list[list[str]]:
    """Paired per-sample gain of the agentic arm over the matched-budget control
    arm (Agentic − Control on the same protein). This is the isolated value of the
    LLM's reasoning: both arms had the same budget, pipeline and selection rule, so
    a positive, consistent gain here is what separates 'the agent reasons well'
    from 'trying N candidates and keeping the best helps'."""
    rows = []
    for key, label in METRIC_NAMES.items():
        stats = distribution_stats(
            [
                (r.get("agent_vs_control_gain") or {}).get(key)
                for r in sample_reports
            ]
        )
        if stats is None:
            continue
        rows.append(
            [
                label,
                f"{stats['mean']:+.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:+.4f}",
                f"{stats['max']:+.4f}",
            ]
        )
    validity_gain_stats = distribution_stats(
        [
            r.get("best_validity_score") - r.get("control_validity_score")
            for r in sample_reports
            if isinstance(r.get("best_validity_score"), (int, float))
            and isinstance(r.get("control_validity_score"), (int, float))
        ]
    )
    if validity_gain_stats is not None:
        rows.append(
            [
                "Best Validity Score (lower=better; negative = agentic more plausible)",
                f"{validity_gain_stats['mean']:+.4f}",
                f"{validity_gain_stats['std']:.4f}",
                f"{validity_gain_stats['min']:+.4f}",
                f"{validity_gain_stats['max']:+.4f}",
            ]
        )
    return rows


def _format_oracle_gap_rows(payload: dict) -> list[list[str]]:
    """Per-metric gap between the validity-selected Agentic Best and the oracle
    (best candidate the agent actually generated). The gap is quality the run left
    on the table purely to imperfect selection — it is reachable with better
    selection alone, no new candidate needed."""
    recon = payload.get("recon_averages", {})
    oracle = payload.get("oracle_averages")
    if not oracle:
        return []
    rows = []
    for key, label in METRIC_NAMES.items():
        rc = recon.get(key, 0.0)
        orc = oracle.get(key, 0.0)
        rows.append([label, f"{rc:.4f}", f"{orc:.4f}", f"{orc - rc:+.4f}"])
    return rows


def _format_distribution_rows(sample_reports: list[dict]) -> list[list[str]]:
    """Per-metric mean/std/min/median/max across samples — the single-glance
    summary that scales to n~100, where a full per-sample table doesn't."""
    rows = []
    for key, label in METRIC_NAMES.items():
        stats = distribution_stats(
            [r.get("recon_metrics", {}).get(key) for r in sample_reports]
        )
        if stats is None:
            continue
        rows.append(
            [
                label,
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['median']:.4f}",
                f"{stats['max']:.4f}",
            ]
        )
    validity_stats = distribution_stats(
        [r.get("best_validity_score") for r in sample_reports]
    )
    if validity_stats is not None:
        rows.append(
            [
                "Best Validity Score (junction+overlap blend, lower=better)",
                f"{validity_stats['mean']:.4f}",
                f"{validity_stats['std']:.4f}",
                f"{validity_stats['min']:.4f}",
                f"{validity_stats['median']:.4f}",
                f"{validity_stats['max']:.4f}",
            ]
        )
    return rows


def _format_cost_rows(payload: dict) -> list[list[str]]:
    """Cost/efficiency and completion accounting — the axis quality alone can't
    justify the agentic approach on. Only populated for agentic runs (the cost
    summary is attached by evaluation/runner.run_agentic)."""
    cost = payload.get("cost_summary")
    if not cost:
        return []
    n = payload.get("sample_count", 0) or 0
    rows = [
        ["Samples", str(n)],
        ["Completed (clean permutation produced)", f"{cost.get('completed_samples', 0)}/{n}"],
        [
            "True order recoverable (ordering metrics valid)",
            f"{cost.get('true_order_recovered', 0)}/{n}",
        ],
        ["LLM lever-choice failures (fell back to defaults)", str(cost.get("llm_failures", 0))],
        ["Total LLM calls", str(cost.get("total_llm_calls", 0))],
        ["Avg LLM calls / sample", f"{cost.get('avg_llm_calls_per_sample', 0):.2f}"],
        ["Total LLM tokens", str(cost.get("total_llm_tokens", 0))],
        ["Avg LLM tokens / sample", f"{cost.get('avg_llm_tokens_per_sample', 0):.0f}"],
        ["Avg wall-clock / sample (total)", _format_duration(cost.get("avg_seconds_per_sample"))],
    ]
    if cost.get("control_enabled"):
        rows.extend(
            [
                ["— Matched-budget control arm (no LLM) —", ""],
                ["Control lever policy", str(cost.get("control_policy", "random"))],
                ["Control LLM calls", "0"],
                [
                    "Avg wall-clock / sample — agentic arm",
                    _format_duration(cost.get("avg_agentic_seconds_per_sample")),
                ],
                [
                    "Avg wall-clock / sample — control arm",
                    _format_duration(cost.get("avg_control_seconds_per_sample")),
                ],
            ]
        )
    return rows


def _format_validity_concordance_rows(sample_reports: list[dict]) -> list[list[str]]:
    """Does the validity selection signal actually track reconstruction quality?
    For each sample, pools the (validity_score, true-metric) pairs across that
    sample's iterations and measures rank concordance — the fraction of
    within-sample candidate pairs where the lower-validity candidate is also the
    higher-quality one. 0.50 means the signal is no better than a coin flip at
    picking the better of two candidates it actually saw."""
    rows = []
    for quality_key, quality_label in (
        ("kendall_tau", "Kendall Tau"),
        ("adjacent_pair_acc", "Adjacent Pair Accuracy"),
    ):
        pairs = []
        for report in sample_reports:
            history = report.get("iteration_history", [])
            for record in history:
                validity = record.get("validity_score")
                quality = record.get("metrics", {}).get(quality_key)
                pairs.append((validity, quality))
        concordance, comparable = rank_concordance(pairs)
        if concordance is None:
            continue
        rows.append(
            [
                quality_label,
                f"{concordance:.3f}",
                str(comparable),
            ]
        )
    return rows


def _select_display_samples(
    sample_reports: list[dict], limit: int = 30, head: int = 15, tail: int = 5
) -> tuple[list[dict], int]:
    """Returns (rows_to_show, hidden_count). Below `limit` samples, shows all
    of them; beyond that, shows the first `head` and last `tail` so the
    markdown report stays scannable at n~100 while samples.jsonl keeps the
    full record and the PDF appendix keeps the full table."""
    if len(sample_reports) <= limit:
        return sample_reports, 0
    displayed = sample_reports[:head] + sample_reports[-tail:]
    return displayed, len(sample_reports) - len(displayed)


def print_run_header(title: str, config_snapshot: dict) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print("Configuration")
    print("-" * 13)
    print(f"  Device: {config_snapshot['misc']['device']}")
    print(f"  Seed: {config_snapshot['misc']['seed']}")
    print(
        f"  Dataset: {config_snapshot['data'].get('organism_display_name') or config_snapshot['data'].get('organism')}"
    )
    print(f"  Test Samples: {config_snapshot['data']['test_samples']}")
    print(f"  Replica Count: {config_snapshot['data']['replica_count']}")
    print(
        f"  Missed Cleavage Ratio: {config_snapshot['data']['missed_cleavage_ratio']}"
    )
    print(f"  MLM Model: {config_snapshot['mlm_model']['name']}")
    print(f"  MLM Type: {config_snapshot['mlm_model']['type']}")
    print(f"  MLM Batch Size: {config_snapshot['mlm_model']['batch_size']}")
    print(f"  MLM Max Length: {config_snapshot['mlm_model']['max_length']}")
    print(f"  Beam Width: {config_snapshot['search']['default_beam_width']}")
    print(f"  Junction Window: {config_snapshot['search']['default_junction_window']}")
    print(f"  Validity Model: {config_snapshot['validity_model']['name']}")
    print(f"  Max Iterations: {config_snapshot['search']['max_iterations']}")
    method = config_snapshot["run"].get("method")
    print(f"  Run Method: {method}")
    if method != "sequential":
        print(f"  Calling Mode: {config_snapshot['run'].get('calling_mode')}")
    iter1_mode = (
        "Deterministic — config defaults (no LLM call)"
        if _iteration1_is_deterministic(config_snapshot)
        else "LLM (genuine agent decision)"
    )
    print(f"  Iteration 1 Mode: {iter1_mode}")
    print(f"  Early Stop Patience: {config_snapshot['search']['early_stop_patience']}")
    print(f"  LLM Model: {config_snapshot['llm_model']['name']}")
    print()


def print_sample_result(index: int, sample_report: dict) -> None:
    print(f"Sample {index}")
    print("-" * 7)
    print(f"  Target:         {sample_report['target']}")
    print(f"  Reconstruction: {sample_report['reconstruction']}")
    if sample_report.get("best_iteration") is not None:
        score = sample_report.get("best_validity_score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        print(f"  Best iteration:  {sample_report['best_iteration']} ({score_text})")
    print(f"  Shuffled order:  {sample_report['baseline_order']}")
    print(f"  Reconstructed:   {sample_report['order']}")
    if sample_report.get("num_pruned") is not None:
        print(
            f"  Filter: {sample_report['num_pruned']}/{sample_report['total_junctions']} junctions pruned ({_format_optional_float(sample_report['pruned_pct'])}%)"
        )
    if sample_report.get("graph") is not None:
        graph = sample_report["graph"]
        print(
            f"  Graph: {graph['num_confirmed_adjacencies']} confirmed adjacencies, {len(graph['unscored_junctions'])} pairs pending scoring"
        )
    print("  Baseline metrics")
    print(_format_metric_block(sample_report["baseline_metrics"]))
    print("  Reconstruction metrics")
    print(_format_metric_block(sample_report["recon_metrics"]))
    print()


def print_summary(
    baseline_summary: dict, recon_summary: dict, sample_count: int
) -> None:
    print(f"Average Results ({sample_count} Samples) — Shuffled vs Reconstructed")
    print("-" * 60)
    print_comparison(baseline_summary, recon_summary, sample_count)


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def _format_duration(seconds) -> str:
    if not isinstance(seconds, (int, float)):
        return "N/A"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_per_sample_rows(sample_reports: list[dict]) -> list[list[str]]:
    rows = []
    for report in sample_reports:
        recon_metrics = report.get("recon_metrics", {})
        exact = "yes" if recon_metrics.get("exact_match") else "no"
        score = report.get("best_validity_score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        num_fragments = len(report.get("order") or [])
        duration = report.get("duration_seconds")
        rows.append(
            [
                str(report.get("index")),
                exact,
                score_text,
                str(report.get("best_iteration", "N/A")),
                str(num_fragments),
                _format_optional_float(report.get("pruned_pct")) + "%",
                str(report.get("graph", {}).get("num_confirmed_adjacencies", "N/A")),
                _format_duration(duration) if duration is not None else "N/A",
            ]
        )
    return rows


def list_run_artifacts(run_dir: Path) -> list[str]:
    """Filenames written into a run's results folder, for printing a concrete
    "saved to these files" summary instead of just the directory path."""
    return sorted(p.name for p in run_dir.iterdir() if p.is_file())


def write_run_results(run_name: str, payload: dict) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    run_dir = RESULTS_ROOT / f"{timestamp}_{_sanitize(run_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    summary_path = run_dir / "summary.json"
    details_path = run_dir / "samples.jsonl"
    report_path = run_dir / "report.md"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    sample_reports = payload.get("samples", [])
    with details_path.open("w", encoding="utf-8") as handle:
        for sample in sample_reports:
            handle.write(json.dumps(sample) + "\n")

    config = payload["config"]
    is_sequential = config.get("run", {}).get("method") == "sequential"
    # Full labels for table headers; short labels for the cramped chart legend.
    fp_label = first_pass_label(config)
    best_label = selected_best_label(config)
    fp_short = (
        "Deterministic (fixed)"
        if _iteration1_is_deterministic(config)
        else "Agentic 1st Pass"
    )
    best_short = "Reconstructed" if is_sequential else "Agentic Best"
    has_control = bool(payload.get("control_averages"))
    has_oracle = bool(payload.get("oracle_averages"))

    bar_chart_svg = render_metric_bar_chart_svg(
        payload.get("baseline_averages", {}),
        payload.get("recon_averages", {}),
        payload.get("first_pass_averages"),
        fp_short,
        best_short,
    )
    (run_dir / "metric_comparison.svg").write_text(bar_chart_svg, encoding="utf-8")

    lines = [f"# {payload['run_name']}", ""]
    lines.extend(run_type_summary(config, has_control=has_control, has_oracle=has_oracle))

    duration = payload.get("duration_seconds")
    overview_lines = [
        "## Run Overview",
        f"- Samples evaluated: {payload['sample_count']}",
        f"- Avg junctions pruned: {payload.get('avg_pruned', 0.0):.1f}%",
        f"- Exact matches: {sum(1 for s in sample_reports if s.get('recon_metrics', {}).get('exact_match'))}/{len(sample_reports)}",
        f"- Result folder: `{run_dir.name}`",
    ]
    if duration is not None:
        overview_lines.append(f"- Total run duration: {_format_duration(duration)}")
        if sample_reports:
            overview_lines.append(
                f"- Avg duration per sample: {_format_duration(duration / len(sample_reports))}"
            )
    lines.extend(overview_lines)
    lines.append("")

    lines.extend(
        [
            "## Configuration",
            _markdown_table(["Setting", "Value"], _format_config_rows(config)),
            "",
        ]
    )

    bench_headers, bench_rows = _format_multi_arm_benchmark(payload, fp_label, best_label)
    three_arm_rows = bench_rows
    if bench_rows:
        # Agentic run with a deterministic baseline: headline is the multi-arm
        # comparison the research question asks for (control/oracle columns appear
        # only when those arms were produced).
        title = "## Benchmark: Shuffled Baseline vs. Deterministic vs. Agentic"
        if has_control:
            title += " vs. Control"
        lines.extend(
            [
                title,
                _markdown_table(bench_headers, bench_rows),
                "",
                "![Metric comparison](metric_comparison.svg)",
                "",
            ]
        )
        oracle_gap_rows = _format_oracle_gap_rows(payload)
        if oracle_gap_rows:
            lines.extend(
                [
                    "### Selection Ceiling (Oracle)",
                    "The Oracle column is the best true-metric value reachable by selecting among "
                    "the candidates the agent already generated (it peeks at ground truth, so it is "
                    "a ceiling, not a method). The gap below is quality the run left on the table "
                    "purely to imperfect validity selection — reachable with better selection alone, "
                    "no new candidate generated. A large gap says the bottleneck is the selection "
                    "signal, not the search.",
                    "",
                    _markdown_table(
                        [
                            "Metric",
                            best_label,
                            "Oracle (best generated)",
                            "Gap (Oracle − Agentic)",
                        ],
                        oracle_gap_rows,
                    ),
                    "",
                ]
            )
    else:
        # Sequential (no deterministic-vs-agentic split): two-arm summary.
        lines.extend(
            [
                "## Benchmark Summary",
                _markdown_table(
                    [
                        "Metric",
                        "Shuffled Baseline",
                        best_label,
                        "Delta",
                        "Interpretation",
                        "Direction",
                    ],
                    _format_metric_rows(payload),
                ),
                "",
                "![Metric comparison](metric_comparison.svg)",
                "",
            ]
        )

    if three_arm_rows:
        gain_dist_rows = _format_iteration_gain_distribution_rows(sample_reports)
        if gain_dist_rows:
            lines.extend(
                [
                    f"## Agentic vs. Deterministic (paired, per sample, n={len(sample_reports)})",
                    "Per-sample gain of the agentic result over the deterministic best-fixed "
                    "baseline (Agentic − Deterministic on the same protein). Mean is the average "
                    "improvement; std dev shows how consistently the agent helps vs. swings the "
                    "other way on individual samples. For a significance claim on n samples, run a "
                    "paired Wilcoxon signed-rank test on these per-sample gains.",
                    "",
                    _markdown_table(
                        ["Metric", "Mean Gain", "Std Dev", "Min Gain", "Max Gain"],
                        gain_dist_rows,
                    ),
                    "",
                ]
            )

    if has_control:
        control_gain_rows = _format_agent_vs_control_distribution_rows(sample_reports)
        if control_gain_rows:
            lines.extend(
                [
                    f"## Agentic vs. Control (paired, matched budget, n={len(sample_reports)})",
                    "**The isolated value of the LLM's reasoning.** Per-sample gain of the agentic "
                    "arm over the non-LLM control arm (Agentic − Control on the same protein), where "
                    "both arms ran the same iteration budget, the same fixed tool pipeline and the "
                    "same best-validity selection — only the lever *source* differed (LLM vs a "
                    f"{payload.get('control_policy', 'random')} policy). A plain Agentic − "
                    "Deterministic gain conflates 'the agent reasons well' with 'trying several "
                    "candidates and keeping the best helps'; this comparison holds the budget and "
                    "selection fixed, so a positive, consistent gain here is attributable to the "
                    "LLM. Run a paired Wilcoxon signed-rank test on these per-sample gains for the "
                    "significance claim.",
                    "",
                    _markdown_table(
                        ["Metric", "Mean Gain", "Std Dev", "Min Gain", "Max Gain"],
                        control_gain_rows,
                    ),
                    "",
                ]
            )

    if sample_reports:
        dist_rows = _format_distribution_rows(sample_reports)
        if dist_rows:
            lines.extend(
                [
                    f"## Distribution Summary (n={len(sample_reports)} samples)",
                    "The at-a-glance view for larger runs — read this before the per-sample table.",
                    "",
                    _markdown_table(
                        ["Metric", "Mean", "Std Dev", "Min", "Median", "Max"],
                        dist_rows,
                    ),
                    "",
                ]
            )

    concordance_rows = _format_validity_concordance_rows(sample_reports)
    if concordance_rows:
        lines.extend(
            [
                "## Validity Signal Concordance",
                "Whether the validity score used to *select* the winning candidate actually "
                "tracks true reconstruction quality, measured within each sample across the "
                "iterations it tried. 0.50 = no better than chance at picking the better of two "
                "candidates; higher is better. This is the trust check for the selection signal — "
                "if it is near 0.50, a better candidate the agent generated would not reliably be "
                "the one kept.",
                "",
                _markdown_table(
                    ["Quality metric compared against", "Concordance", "Comparable pairs"],
                    concordance_rows,
                ),
                "",
            ]
        )

    cost_rows = _format_cost_rows(payload)
    if cost_rows:
        lines.extend(
            [
                "## Cost, Efficiency & Completion",
                "The non-quality axis the agentic approach must also justify itself on: LLM "
                "calls, tokens, wall-clock, and how often it produced a usable result.",
                "",
                _markdown_table(["Measure", "Value"], cost_rows),
                "",
            ]
        )

    if sample_reports:
        display_rows, hidden_count = _select_display_samples(sample_reports)
        table_title = "## Per-Sample Results"
        if hidden_count:
            table_title += (
                f" (showing first {len(display_rows) - min(5, len(display_rows))} and "
                f"last {min(5, len(display_rows))} of {len(sample_reports)}; "
                f"full table in `samples.jsonl`)"
            )
        lines.extend(
            [
                table_title,
                _markdown_table(
                    [
                        "Sample",
                        "Exact Match",
                        "Best Validity Score",
                        "Best Iteration",
                        "Fragments Placed",
                        "Junctions Pruned",
                        "Confirmed Adjacencies",
                        "Duration",
                    ],
                    _format_per_sample_rows(display_rows),
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Quick Read",
            "- Higher is better for every metric in the current set.",
            "- Metrics: Exact Match (binary floor); Sequence Similarity (the one soft string "
            "metric); Adjacent Pair Accuracy (fraction of true fragment adjacencies preserved, "
            "the primary ordering metric); Longest Correct Run (longest contiguous correctly-"
            "ordered block, partial-assembly credit); Kendall Tau (global ordering correlation, "
            "0 = random, 1 = perfect, -1 = reversed).",
            "- Ordering metrics are NaN (skipped in the averages) for any sample whose fragments "
            "do not tile the target; the count of usable samples is in Cost, Efficiency & Completion.",
            "- A positive delta means the reconstruction improved over the shuffled baseline.",
            "- Each entry in samples.jsonl includes iteration_history with per-iteration lever_values and changed_levers for auditability.",
            "- The validity score is the junction+overlap blended plausibility signal (lower = "
            "better); it measures plausibility, not exact-match correctness. Its trustworthiness "
            "is quantified in Validity Signal Concordance above.",
            "- Junction-scorer ranking quality is measured separately and search-independently via "
            "`python -m evaluation.junction_ranking`.",
            "- Use this report for side-by-side benchmarking; the raw per-sample data is in `samples.jsonl`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return run_dir
