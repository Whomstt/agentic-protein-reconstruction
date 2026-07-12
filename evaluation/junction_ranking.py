"""Search-independent diagnostic for the pipeline's core assumption: that the
pLM junction scorer ranks the true successor fragment above the wrong ones.

Every metric in evaluation/metrics.py is measured *after* greedy/beam search, so
it entangles scorer quality with search dynamics, trypsin constraints, and
overlap-graph confirmations. This module scores a dense junction matrix over all
ordered fragment pairs and asks, for each true adjacency i->s, where s ranks
among all candidate successors of i — with no search involved.

Run directly:

    python -m evaluation.junction_ranking
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from algorithms.score_junctions import score_junctions
from config import cfg
from evaluation.metrics import junction_ranking_stats, nanmean, recover_true_order
from evaluation.reporting import RESULTS_ROOT, _markdown_table, _sanitize
from evaluation.runner import _load_test_samples
from models.memory import free_gpu_memory


def _junction_window() -> int:
    return int(cfg["search"]["default_levers"]["junction_window"])


def evaluate_junction_ranking(window: int | None = None) -> dict:
    """Scores a dense junction matrix per sample and measures how well the raw
    pLM ranks true successors, independent of search. Returns an aggregate dict."""
    window = _junction_window() if window is None else int(window)
    samples = _load_test_samples()

    per_sample = []
    for i, sample in enumerate(samples, 1):
        target = sample.get(
            cfg["data"]["active_target_key"], sample.get("target_reconstruction")
        )
        fragment_samples = sample.get("fragment_samples") or [sample["fragments"]]
        fragments = fragment_samples[0]
        true_order = recover_true_order(target, fragments)
        if true_order is None:
            per_sample.append(
                {"index": i, "num_fragments": len(fragments), "recovered": False}
            )
            print(
                f"  Sample {i}/{len(samples)}: true order not recoverable "
                f"({len(fragments)} fragments) — skipped"
            )
            continue

        # Dense matrix over all ordered pairs (no confirmed-junction sentinel),
        # i.e. the raw scorer with nothing from the overlap graph mixed in.
        matrix = score_junctions(fragments, window=window)
        stats = junction_ranking_stats(matrix, true_order, len(fragments))
        stats.update({"index": i, "num_fragments": len(fragments), "recovered": True})
        per_sample.append(stats)
        print(
            f"  Sample {i}/{len(samples)}: top1={stats['top1_acc']:.3f} "
            f"top3={stats['top3_acc']:.3f} mrr={stats['mrr']:.3f} "
            f"over {stats['num_junctions']} true junctions"
        )
        free_gpu_memory()

    recovered = [s for s in per_sample if s.get("recovered")]
    aggregate = {
        "window": window,
        "num_samples": len(samples),
        "num_recovered": len(recovered),
        "top1_acc": nanmean([s.get("top1_acc") for s in recovered]),
        "top3_acc": nanmean([s.get("top3_acc") for s in recovered]),
        "mrr": nanmean([s.get("mrr") for s in recovered]),
        "total_junctions": sum(s.get("num_junctions", 0) for s in recovered),
    }
    return {"aggregate": aggregate, "per_sample": per_sample}


def _write_report(result: dict, organism: str, duration: float):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    run_dir = RESULTS_ROOT / f"{timestamp}_{_sanitize('junction_ranking')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    agg = result["aggregate"]
    rows = [
        ["Junction masking window", str(agg["window"])],
        ["Samples", str(agg["num_samples"])],
        ["Samples with recoverable true order", str(agg["num_recovered"])],
        ["True junctions scored", str(agg["total_junctions"])],
        ["Top-1 successor accuracy", f"{agg['top1_acc']:.4f}"],
        ["Top-3 successor accuracy", f"{agg['top3_acc']:.4f}"],
        ["Mean reciprocal rank", f"{agg['mrr']:.4f}"],
    ]
    lines = [
        "# Junction Scorer Ranking (search-independent)",
        "",
        "How well the raw pLM junction scorer ranks the true successor fragment "
        "above the wrong ones, before any search or constraint is applied. Top-1 "
        "accuracy is the fraction of true junctions where the true successor is "
        "the single highest-scoring candidate; MRR is the mean of 1/rank. A random "
        f"scorer over ~k candidates would score ~1/k. Dataset: {organism}.",
        "",
        _markdown_table(["Measurement", "Value"], rows),
        "",
        f"Total duration: {duration:.1f}s.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return run_dir


def main():
    started = time.time()
    organism = cfg["data"].get("organism_display_name") or cfg["data"].get("organism")
    window = _junction_window()
    print(
        f"\nJunction Scorer Ranking — {organism}, window={window} "
        f"({cfg['mlm_model']['profile']})\n" + "-" * 60
    )
    result = evaluate_junction_ranking(window=window)
    duration = time.time() - started
    agg = result["aggregate"]
    print("-" * 60)
    print(
        f"Aggregate over {agg['num_recovered']}/{agg['num_samples']} samples "
        f"({agg['total_junctions']} true junctions):"
    )
    print(f"  Top-1 successor accuracy: {agg['top1_acc']:.4f}")
    print(f"  Top-3 successor accuracy: {agg['top3_acc']:.4f}")
    print(f"  Mean reciprocal rank:     {agg['mrr']:.4f}")
    run_dir = _write_report(result, organism, duration)
    print(f"\nSaved to {run_dir}\n")


if __name__ == "__main__":
    main()
