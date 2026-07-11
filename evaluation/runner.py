"""Single source of truth for running an evaluation over the active test
split: both the deterministic pipeline (run_sequential) and the LLM-driven
iterative agent (run_agentic). main.py, evaluation/sequential.py, and
evaluation/agentic.py all call into this module instead of duplicating the
sample loop, progress display, and reporting logic.
"""

from __future__ import annotations

import json
import os
import random
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError

from config import cfg
from evaluation.metrics import METRIC_NAMES, compute_all
from evaluation.reporting import (
    build_config_snapshot,
    list_run_artifacts,
    print_run_header,
    print_summary,
    write_run_results,
)
from models.memory import free_gpu_memory, log_memory

load_dotenv(override=True)

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RESET = "\033[0m"

WRAP_WIDTH = 96
PROGRESS_WIDTH = 28


def _wrap(text: str, indent: str) -> str:
    wrapper = textwrap.TextWrapper(
        width=WRAP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(wrapper.wrap(text)) if text else indent


def _format_args(args: dict) -> str:
    if not args:
        return ""
    parts = [f"{k}={v!r}" for k, v in args.items()]
    return ", ".join(parts)


def _preview(value, length=90) -> str:
    text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= length else text[:length] + "..."


def _format_duration(seconds) -> str:
    if seconds is None:
        return "n/a"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _progress_bar(done: int, total: int, width: int = PROGRESS_WIDTH) -> str:
    filled = width if total <= 0 else round(width * done / total)
    return "█" * filled + "░" * (width - filled)


def _print_sample_progress_header(index: int, total: int, sample_durations: list[float]):
    completed = index - 1
    bar = _progress_bar(completed, total)
    elapsed = sum(sample_durations)
    avg = elapsed / completed if completed else None
    remaining = total - completed
    eta = avg * remaining if avg is not None else None

    print(f"\n{BOLD}{BLUE}{'═' * 60}{RESET}")
    print(
        f"{BOLD}{BLUE}  Sample {index}/{total}{RESET}  {BLUE}[{bar}]{RESET}  "
        f"{DIM}elapsed {_format_duration(elapsed)} │ ETA {_format_duration(eta)}{RESET}"
    )
    print(f"{BOLD}{BLUE}{'═' * 60}{RESET}")


def make_event_printer():
    """Returns an on_event(kind, payload) callback that live-prints agent reasoning."""

    def on_event(kind, payload):
        if kind == "iteration_start":
            i, n = payload["iteration"], payload["max_iterations"]
            print(f"\n{BOLD}{CYAN}┌─ Iteration {i}/{n} {'─' * 40}{RESET}")

        elif kind == "thought":
            print(f"{CYAN}│{RESET} {DIM}thinking:{RESET}")
            print(_wrap(payload, indent=f"{CYAN}│{RESET}   "))

        elif kind == "tool_call":
            name = payload["name"]
            args_str = _format_args(payload.get("args", {}))
            print(f"{CYAN}│{RESET} {MAGENTA}→ {name}{RESET}({DIM}{args_str}{RESET})")

        elif kind == "tool_result":
            name = payload["name"]
            content_preview = _preview(payload.get("content"))
            print(f"{CYAN}│{RESET}   {DIM}⤷ {content_preview}{RESET}")

        elif kind == "iteration_end":
            score = payload.get("validity_score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            summary = payload.get("strategy_summary", "")
            print(
                f"{CYAN}└─{RESET} {BOLD}score={score_text}{RESET} {DIM}| {summary}{RESET}"
            )

    return on_event


def _load_test_samples() -> list[dict]:
    """Loads the deduped fragmented pool for the active organism, shuffles it
    (using the global seed set once in config.py), and takes the first
    data.test_samples records."""
    pool_path = cfg["data"]["active_fragmented_split"]
    # Read the raw lines (cheap strings) and shuffle those, then JSON-parse only
    # the records we actually keep. Parsing every line into nested dicts/lists up
    # front was the single largest RAM spike at high replica_count (the pool file
    # is ~20x bigger at r=100), even though only test_samples records are used.
    # Shuffling the line list with the same seed selects the identical records
    # the old "parse-all then shuffle then slice" path would have.
    with open(pool_path) as f:
        lines = [line for line in f if line.strip()]

    random.shuffle(lines)

    limit = cfg["data"].get("test_samples")
    selected = lines[:limit] if limit else lines
    return [json.loads(line) for line in selected]


def _print_fragment_input(sample: dict, fragment_samples, fragments) -> None:
    if sample.get("fragment_samples"):
        print(
            f"\n{BOLD}  Input: {len(fragment_samples)} digestion sample(s), {len(fragments)} unique fragments{RESET}"
        )
    else:
        print(f"\n{BOLD}  Input: {len(fragments)} fragments{RESET}")
    for idx, frag in enumerate(fragments):
        preview = frag[:30] + "..." if len(frag) > 30 else frag
        print(f"{DIM}  [{idx}] {preview} ({len(frag)} aa){RESET}")


LEVER_LABELS = {
    "junction_window": "junction_window",
    "search_mode": "search_mode",
    "beam_width": "beam_width",
    "edge_mode": "edge_mode",
    "confirmed_bonus": "confirmed_bonus",
}


def _format_lever_values(lever_values: dict) -> str:
    """Renders the fully resolved lever values for an iteration (defaults
    filled in, not just whatever the agent explicitly passed) so the console
    always shows the actual params used, even on iteration 1 / the baseline
    pass where the agent typically relies on defaults."""
    return ", ".join(
        f"{label}={lever_values[key]}"
        for key, label in LEVER_LABELS.items()
        if key in lever_values
    )


def _print_saved_artifacts(run_dir: Path) -> None:
    print(f"\n{BOLD}  Saved run artifacts to {run_dir}{RESET}")
    for name in list_run_artifacts(run_dir):
        print(f"{DIM}    - {name}{RESET}")


def _print_result_block(target: str, reconstruction: str, validity_score=None) -> None:
    print(f"\n{'─' * 60}")
    print(f"{BOLD}  Result{RESET}")
    print(f"{'─' * 60}")
    if reconstruction:
        match = target == reconstruction
        status = f"{GREEN}✓ Exact match{RESET}" if match else f"{YELLOW}✗ Mismatch{RESET}"
        print(
            f"  Target:        {DIM}{target[:70]}{'...' if len(target) > 70 else ''}{RESET}"
        )
        print(
            f"  Reconstructed: {DIM}{reconstruction[:70]}{'...' if len(reconstruction) > 70 else ''}{RESET}"
        )
        if isinstance(validity_score, (int, float)):
            print(f"  Validity score: {DIM}{validity_score:.4f}{RESET}")
        print(f"  {status}")
    else:
        print(f"  {YELLOW}No reconstruction produced{RESET}")


def run_sequential() -> Path | None:
    """Deterministic pipeline evaluation (no LLM): trypsin_filter -> overlap_graph
    -> score_junctions -> beam_order, run once per test sample."""
    from evaluation.pipeline import reconstruct

    samples = _load_test_samples()
    n = len(samples)
    config_snapshot = build_config_snapshot(cfg)
    run_name = (
        f"Sequential Evaluation ({cfg['data']['organism_display_name']}, "
        f"{cfg['mlm_model']['profile']}, r{cfg['data']['replica_count']})"
    )
    print_run_header(f"{run_name} ({n} Samples)", config_snapshot)

    baseline_summary = {k: [] for k in METRIC_NAMES}
    recon_summary = {k: [] for k in METRIC_NAMES}
    sample_reports = []
    sample_durations = []
    run_started = time.time()

    for i, sample in enumerate(samples, 1):
        sample_start = time.time()
        _print_sample_progress_header(i, n, sample_durations)

        target = sample.get(
            cfg["data"]["active_target_key"], sample.get("target_reconstruction")
        )
        fragment_samples = sample.get("fragment_samples") or [sample["fragments"]]
        fragments = fragment_samples[0]
        _print_fragment_input(sample, fragment_samples, fragments)

        reconstruction, order, constraints, graph = reconstruct(fragment_samples)
        _print_result_block(target, reconstruction)

        baseline_order = list(range(len(fragments)))
        random.shuffle(baseline_order)
        baseline_recon = "".join(fragments[idx] for idx in baseline_order)
        baseline_metrics = compute_all(target, baseline_recon, fragments, baseline_order)
        recon_metrics = compute_all(target, reconstruction, fragments, order)

        for k in METRIC_NAMES:
            baseline_summary[k].append(baseline_metrics[k])
            recon_summary[k].append(recon_metrics[k])

        sample_duration = time.time() - sample_start
        sample_durations.append(sample_duration)

        total_junctions = len(fragments) * (len(fragments) - 1)
        pruned = len(constraints["impossible_junctions"])
        sample_reports.append(
            {
                "index": i,
                "target": target,
                "reconstruction": reconstruction,
                "baseline_order": baseline_order,
                "order": order,
                "baseline_metrics": baseline_metrics,
                "recon_metrics": recon_metrics,
                "num_pruned": pruned,
                "total_junctions": total_junctions,
                "pruned_pct": (pruned / total_junctions * 100) if total_junctions else 0.0,
                "graph": {
                    "num_confirmed_adjacencies": len(graph["confirmed_adjacencies"]),
                    "unscored_junctions": graph["unscored_junctions"],
                },
                "duration_seconds": sample_duration,
            }
        )
        print(f"\n{DIM}  Sample {i}/{n} finished in {_format_duration(sample_duration)}{RESET}")
        free_gpu_memory()
        log_memory(f"after sample {i}/{n}")

    if not samples:
        print(f"\n{YELLOW}No test samples found for the active dataset.{RESET}")
        return None

    total_duration = time.time() - run_started
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Run Summary{RESET}")
    print(f"{'─' * 60}")
    print_summary(baseline_summary, recon_summary, n)
    print(
        f"\n{DIM}  Total duration: {_format_duration(total_duration)} "
        f"(avg {_format_duration(total_duration / n)}/sample){RESET}"
    )

    baseline_averages = {k: sum(v) / n for k, v in baseline_summary.items()}
    recon_averages = {k: sum(v) / n for k, v in recon_summary.items()}
    delta = {k: recon_averages[k] - baseline_averages[k] for k in METRIC_NAMES}
    avg_pruned = sum(r["pruned_pct"] for r in sample_reports) / n

    run_payload = {
        "run_name": run_name,
        "config": config_snapshot,
        "sample_count": n,
        "avg_pruned": avg_pruned,
        "baseline_averages": baseline_averages,
        "recon_averages": recon_averages,
        "delta": delta,
        "samples": sample_reports,
        "duration_seconds": total_duration,
    }
    run_dir = write_run_results("sequential", run_payload)
    _print_saved_artifacts(run_dir)
    print(f"{'═' * 60}\n")
    return run_dir


def run_agentic() -> Path | None:
    """LLM-driven iterative agent evaluation, run once per test sample."""
    from agents.iterative_runner import run_iterative_reconstruction
    from agents.react_agent import build_agent

    llm_config = cfg["llm_model"]
    api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "").strip().strip("'\"")
    if llm_config.get("kind") == "microsoft_foundry":
        endpoint_env = llm_config["endpoint_env"]
        endpoint = os.environ.get(endpoint_env, "").strip().strip("'\"")
        if not endpoint:
            print(f"\n{YELLOW}Set {endpoint_env} in your environment or .env file.{RESET}")
            return None
        if llm_config.get("auth_mode", "auto") == "api_key" and not api_key:
            print(f"\n{YELLOW}Set {api_key_env} in your environment or .env file.{RESET}")
            return None
    elif not api_key:
        print(f"\n{YELLOW}Set {api_key_env} in your environment or .env file.{RESET}")
        return None

    if api_key:
        os.environ[api_key_env] = api_key

    try:
        agent = build_agent()
    except AuthenticationError:
        print(
            f"\n{YELLOW}{api_key_env} was rejected by the selected LLM provider. Check that the key is current and copied exactly.{RESET}"
        )
        return None

    samples = _load_test_samples()
    n = len(samples)
    config_snapshot = build_config_snapshot(cfg)
    run_name = (
        f"Agentic Evaluation ({cfg['data']['organism_display_name']}, "
        f"{cfg['mlm_model']['profile']}, r{cfg['data']['replica_count']})"
    )
    print_run_header(f"{run_name} ({n} Samples)", config_snapshot)

    baseline_summary = {k: [] for k in METRIC_NAMES}
    first_pass_summary = {k: [] for k in METRIC_NAMES}
    recon_summary = {k: [] for k in METRIC_NAMES}
    sample_reports = []
    sample_durations = []
    run_started = time.time()

    for i, sample in enumerate(samples, 1):
        sample_start = time.time()
        _print_sample_progress_header(i, n, sample_durations)

        target = sample.get(
            cfg["data"]["active_target_key"], sample.get("target_reconstruction")
        )
        fragment_samples = sample.get("fragment_samples") or [sample["fragments"]]
        fragments = fragment_samples[0]
        _print_fragment_input(sample, fragment_samples, fragments)

        print(f"\n{BOLD}{'─' * 60}{RESET}")
        print(f"{BOLD}  Live reasoning{RESET}")
        print(f"{'─' * 60}")

        try:
            run_result = run_iterative_reconstruction(
                agent, fragments, fragment_samples, on_event=make_event_printer()
            )
        except AuthenticationError:
            print(
                f"\n{YELLOW}The selected LLM rejected the API key during the agent run. Update {api_key_env} and try again.{RESET}"
            )
            return None

        best_record = run_result.get("best_record", {})
        first_record = run_result.get("first_record", {})
        iteration_history = run_result.get("iteration_history", [])
        reconstruction = best_record.get("reconstruction", "")
        order = best_record.get("order", [])
        validity_score = best_record.get("validity_score")
        state_snapshot = run_result.get("state", {})

        print(f"\n{'─' * 60}")
        print(f"{BOLD}  Iteration overview{RESET}")
        print(f"{'─' * 60}")
        best_iter = best_record.get("iteration")
        first_iter = first_record.get("iteration")

        # Attach each iteration's true reconstruction metrics (vs. the target) to
        # its history record. Cheap (string metrics) and makes runs fully
        # auditable per iteration; it also lets search.improvement_margin be
        # tuned offline against actual quality instead of only validity.
        for record in iteration_history:
            record["metrics"] = compute_all(
                target,
                record.get("reconstruction", ""),
                fragments,
                record.get("order", []),
            )

        for record in iteration_history:
            score = record.get("validity_score")
            summary = record.get("strategy_summary", "")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            marker = " *" if record.get("iteration") == best_iter else "  "
            baseline_tag = (
                f" {DIM}(baseline){RESET}"
                if record.get("iteration") == first_iter
                else ""
            )
            print(
                f"{marker}Iteration {record['iteration']}: score={score_text} | {summary}{baseline_tag}"
            )
            lever_text = _format_lever_values(record.get("lever_values", {}))
            if lever_text:
                print(f"{DIM}    params: {lever_text}{RESET}")

        _print_result_block(target, reconstruction, validity_score)

        baseline_order = list(range(len(fragments)))
        random.shuffle(baseline_order)
        baseline_recon = "".join(fragments[idx] for idx in baseline_order)
        baseline_metrics = compute_all(target, baseline_recon, fragments, baseline_order)
        recon_metrics = compute_all(target, reconstruction, fragments, order)

        # First pass = iteration 1, i.e. what a single-shot (non-iterative)
        # agent call would have produced. Comparing it to the best iteration
        # isolates the value added by the iterative refinement loop itself.
        first_pass_reconstruction = first_record.get("reconstruction", "")
        first_pass_order = first_record.get("order", [])
        first_pass_metrics = compute_all(
            target, first_pass_reconstruction, fragments, first_pass_order
        )
        iteration_gain = {
            k: recon_metrics[k] - first_pass_metrics[k] for k in METRIC_NAMES
        }

        for k in METRIC_NAMES:
            baseline_summary[k].append(baseline_metrics[k])
            first_pass_summary[k].append(first_pass_metrics[k])
            recon_summary[k].append(recon_metrics[k])

        print(f"\n{DIM}  Iteration gain (best iter {best_record.get('iteration')} vs. first pass):{RESET}")
        for k, label in METRIC_NAMES.items():
            gain = iteration_gain[k]
            better = (gain < 0) if k == "norm_edit_distance" else (gain > 0)
            tag = (
                f"{GREEN}better{RESET}"
                if better and gain != 0
                else (f"{YELLOW}worse{RESET}" if gain != 0 else f"{DIM}same{RESET}")
            )
            print(f"{DIM}    {label}: {gain:+.4f} ({tag}{DIM}){RESET}")

        sample_duration = time.time() - sample_start
        sample_durations.append(sample_duration)

        total_junctions = len(fragments) * (len(fragments) - 1)
        sample_reports.append(
            {
                "index": i,
                "target": target,
                "reconstruction": reconstruction,
                "baseline_order": baseline_order,
                "order": order,
                "baseline_metrics": baseline_metrics,
                "recon_metrics": recon_metrics,
                "first_pass_metrics": first_pass_metrics,
                "first_pass_validity_score": first_record.get("validity_score"),
                "iteration_gain": iteration_gain,
                "best_iteration": best_record.get("iteration"),
                "best_validity_score": validity_score,
                "iteration_history": iteration_history,
                "num_pruned": len(state_snapshot.get("impossible_junctions", [])),
                "total_junctions": total_junctions,
                "pruned_pct": (
                    len(state_snapshot.get("impossible_junctions", [])) / total_junctions * 100
                    if total_junctions > 0
                    else 0.0
                ),
                "graph": {
                    "num_confirmed_adjacencies": len(
                        state_snapshot.get("confirmed_adjacencies", [])
                    ),
                    "unscored_junctions": state_snapshot.get("unscored_junctions", []),
                },
                "duration_seconds": sample_duration,
            }
        )
        print(f"\n{DIM}  Sample {i}/{n} finished in {_format_duration(sample_duration)}{RESET}")
        free_gpu_memory()

    if not samples:
        print(f"\n{YELLOW}No test samples found for the active dataset.{RESET}")
        return None

    total_duration = time.time() - run_started
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Run Summary{RESET}")
    print(f"{'─' * 60}")
    print_summary(baseline_summary, recon_summary, n)
    print(
        f"\n{DIM}  Total duration: {_format_duration(total_duration)} "
        f"(avg {_format_duration(total_duration / n)}/sample){RESET}"
    )

    baseline_averages = {k: sum(v) / n for k, v in baseline_summary.items()}
    first_pass_averages = {k: sum(v) / n for k, v in first_pass_summary.items()}
    recon_averages = {k: sum(v) / n for k, v in recon_summary.items()}
    delta = {k: recon_averages[k] - baseline_averages[k] for k in METRIC_NAMES}
    avg_pruned = sum(r["pruned_pct"] for r in sample_reports) / n

    run_payload = {
        "run_name": run_name,
        "config": config_snapshot,
        "sample_count": n,
        "avg_pruned": avg_pruned,
        "baseline_averages": baseline_averages,
        "first_pass_averages": first_pass_averages,
        "recon_averages": recon_averages,
        "delta": delta,
        "samples": sample_reports,
        "duration_seconds": total_duration,
    }
    run_dir = write_run_results("agentic", run_payload)
    _print_saved_artifacts(run_dir)
    print(f"{'═' * 60}\n")
    return run_dir
