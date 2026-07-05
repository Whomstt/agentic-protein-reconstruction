from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evaluation.metrics import METRIC_NAMES, print_comparison

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def build_config_snapshot(cfg: dict) -> dict:
    return {
        "misc": {
            "seed": cfg["misc"].get("seed"),
            "device": cfg["misc"].get("device"),
        },
        "mlm_model": {
            "type": cfg["mlm_model"].get("type"),
            "name": cfg["mlm_model"].get("name"),
            "batch_size": cfg["mlm_model"].get("batch_size"),
            "max_length": cfg["mlm_model"].get("max_length"),
            "beam_size": cfg["mlm_model"].get("beam_size"),
            "junction_window": cfg["mlm_model"].get("junction_window"),
        },
        "validity_model": {
            "name": cfg.get("validity_model", {}).get("name"),
            "batch_size": cfg.get("validity_model", {}).get("batch_size"),
            "max_length": cfg.get("validity_model", {}).get("max_length"),
        },
        "llm_model": {
            "name": cfg["llm_model"].get("name"),
            "temperature": cfg["llm_model"].get("temperature"),
        },
        "search": {
            "max_iterations": cfg.get("search", {}).get("max_iterations"),
            "validity_threshold": cfg.get("search", {}).get("validity_threshold"),
            "beam_width_step": cfg.get("search", {}).get("beam_width_step"),
        },
        "data": {
            "organism": cfg["data"].get("organism"),
            "organism_display_name": cfg["data"].get("organism_display_name"),
            "test_ratio": cfg["data"].get("test_ratio"),
            "test_samples": cfg["data"].get("test_samples"),
            "replica_count": cfg["data"].get(
                "replica_count", cfg["data"].get("sample_count")
            ),
            "missed_cleavage_ratio": cfg["data"].get("missed_cleavage_ratio"),
            "active_test_split": cfg["data"].get("active_test_split"),
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
    return [
        ["Device", str(config["misc"]["device"])],
        ["Seed", str(config["misc"]["seed"])],
        [
            "Dataset",
            str(
                config["data"].get("organism_display_name")
                or config["data"].get("organism")
            ),
        ],
        ["Test Ratio", str(config["data"]["test_ratio"])],
        ["Test Samples", str(config["data"]["test_samples"])],
        ["Replica Count", str(config["data"]["replica_count"])],
        ["Missed Cleavage Ratio", str(config["data"]["missed_cleavage_ratio"])],
        ["MLM Model", str(config["mlm_model"]["name"])],
        ["MLM Type", str(config["mlm_model"]["type"])],
        ["MLM Batch Size", str(config["mlm_model"]["batch_size"])],
        ["MLM Max Length", str(config["mlm_model"]["max_length"])],
        ["Beam Size", str(config["mlm_model"]["beam_size"])],
        ["Junction Window", str(config["mlm_model"]["junction_window"])],
        ["Validity Model", str(config.get("validity_model", {}).get("name"))],
        ["Max Iterations", str(config.get("search", {}).get("max_iterations"))],
        [
            "Validity Threshold",
            str(config.get("search", {}).get("validity_threshold")),
        ],
        ["Beam Width Step", str(config.get("search", {}).get("beam_width_step"))],
        ["LLM Model", str(config["llm_model"]["name"])],
        ["LLM Temperature", str(config["llm_model"]["temperature"])],
    ]


def _format_metric_rows(payload: dict) -> list[list[str]]:
    rows = []
    for key, label in METRIC_NAMES.items():
        baseline = payload["baseline_averages"][key]
        recon = payload["recon_averages"][key]
        delta = payload["delta"][key]
        better = "up" if key != "norm_edit_distance" else "down"
        if key == "norm_edit_distance":
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
    print(f"  Test Ratio: {config_snapshot['data']['test_ratio']}")
    print(f"  Test Samples: {config_snapshot['data']['test_samples']}")
    print(f"  Replica Count: {config_snapshot['data']['replica_count']}")
    print(
        f"  Missed Cleavage Ratio: {config_snapshot['data']['missed_cleavage_ratio']}"
    )
    print(f"  MLM Model: {config_snapshot['mlm_model']['name']}")
    print(f"  MLM Type: {config_snapshot['mlm_model']['type']}")
    print(f"  MLM Batch Size: {config_snapshot['mlm_model']['batch_size']}")
    print(f"  MLM Max Length: {config_snapshot['mlm_model']['max_length']}")
    print(f"  Beam Size: {config_snapshot['mlm_model']['beam_size']}")
    print(f"  Junction Window: {config_snapshot['mlm_model']['junction_window']}")
    print(f"  Validity Model: {config_snapshot['validity_model']['name']}")
    print(f"  Max Iterations: {config_snapshot['search']['max_iterations']}")
    print(f"  Validity Threshold: {config_snapshot['search']['validity_threshold']}")
    print(f"  Beam Width Step: {config_snapshot['search']['beam_width_step']}")
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


def write_run_results(run_name: str, payload: dict) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / f"{timestamp}_{_sanitize(run_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    summary_path = run_dir / "summary.json"
    details_path = run_dir / "samples.jsonl"
    report_path = run_dir / "report.md"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with details_path.open("w", encoding="utf-8") as handle:
        for sample in payload.get("samples", []):
            handle.write(json.dumps(sample) + "\n")

    lines = [f"# {payload['run_name']}", ""]
    config = payload["config"]
    lines.extend(
        [
            "## Run Overview",
            f"- Samples evaluated: {payload['sample_count']}",
            f"- Avg junctions pruned: {payload.get('avg_pruned', 0.0):.1f}%",
            f"- Result folder: `{run_dir.name}`",
            "",
            "## Configuration",
            _markdown_table(["Setting", "Value"], _format_config_rows(config)),
            "",
            "## Benchmark Summary",
            _markdown_table(
                [
                    "Metric",
                    "Baseline",
                    "Reconstructed",
                    "Delta",
                    "Interpretation",
                    "Direction",
                ],
                _format_metric_rows(payload),
            ),
            "",
            "## Quick Read",
            "- Higher is better for all metrics except normalized edit distance.",
            "- A positive delta means the reconstruction improved over the shuffled baseline.",
            "- Each entry in samples.jsonl includes iteration_history with per-iteration lever_values and changed_levers for auditability.",
            "- Use this report for side-by-side benchmarking; the raw per-sample data is in `samples.jsonl`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return run_dir
