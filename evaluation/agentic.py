import json
import random
from agents.react_agent import build_agent
from evaluation.metrics import (
    METRIC_NAMES,
    compute_all,
)
from evaluation.reporting import (
    build_config_snapshot,
    print_run_header,
    print_sample_result,
    print_summary,
    write_run_results,
)
from config import cfg


def extract_reconstruction(result):
    """Extract reconstruction and order from the agent's final tool call."""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "name") and msg.name == "beam_search":
            content = (
                json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            )
            return content["reconstruction"], content["order"]
    return result["messages"][-1].content, []


test_path = cfg["data"]["active_test_split"]
test_samples = cfg["data"].get("test_samples")
config_snapshot = build_config_snapshot(cfg)
run_name = f"Agentic Evaluation ({cfg['data']['organism_display_name']})"

with open(test_path) as f:
    samples = [json.loads(l) for l in f if l.strip()][:test_samples]

agent = build_agent()

baseline_summary = {k: [] for k in METRIC_NAMES}
recon_summary = {k: [] for k in METRIC_NAMES}

print_run_header(f"{run_name} ({len(samples)} Samples)", config_snapshot)

sample_reports = []
for i, sample in enumerate(samples, 1):
    target = sample.get(
        cfg["data"]["active_target_key"], sample.get("target_reconstruction")
    )
    fragment_samples = sample.get("fragment_samples") or [sample["fragments"]]
    fragments = fragment_samples[0]

    # Baseline: a fresh random permutation of the fragments.
    baseline_order = list(range(len(fragments)))
    random.Random(cfg["misc"].get("seed", 0) + i).shuffle(baseline_order)
    baseline_recon = "".join(fragments[idx] for idx in baseline_order)
    baseline_metrics = compute_all(target, baseline_recon, fragments, baseline_order)

    # Reconstructed: agent output
    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    (
                        f"Reconstruct the protein from these digestion samples: {fragment_samples}"
                        if sample.get("fragment_samples")
                        else f"Reconstruct the protein from these fragments: {fragments}"
                    ),
                )
            ]
        }
    )
    reconstruction, order = extract_reconstruction(result)
    recon_metrics = compute_all(target, reconstruction, fragments, order)

    for k in METRIC_NAMES:
        baseline_summary[k].append(baseline_metrics[k])
        recon_summary[k].append(recon_metrics[k])
    sample_report = {
        "index": i,
        "target": target,
        "reconstruction": reconstruction,
        "baseline_order": baseline_order,
        "order": order,
        "baseline_metrics": baseline_metrics,
        "recon_metrics": recon_metrics,
        "num_pruned": None,
        "total_junctions": None,
        "pruned_pct": None,
        "graph": None,
    }
    sample_reports.append(sample_report)
    print_sample_result(i, sample_report)

if samples:
    n = len(samples)
    baseline_averages = {k: sum(v) / n for k, v in baseline_summary.items()}
    recon_averages = {k: sum(v) / n for k, v in recon_summary.items()}
    delta = {k: recon_averages[k] - baseline_averages[k] for k in METRIC_NAMES}
    print_summary(baseline_summary, recon_summary, n)

    run_payload = {
        "run_name": run_name,
        "config": config_snapshot,
        "sample_count": n,
        "avg_pruned": 0.0,
        "baseline_averages": baseline_averages,
        "recon_averages": recon_averages,
        "delta": delta,
        "samples": sample_reports,
    }
    run_dir = write_run_results("agentic", run_payload)
    print(f"\nSaved run artifacts to {run_dir}")
