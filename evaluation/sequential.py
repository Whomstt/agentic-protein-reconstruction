import json
import random
from algorithms.trypsin_filter import trypsin_filter
from algorithms.overlap_graph import build_overlap_graph
from algorithms.score_junctions import score_junctions
from algorithms.beam_order import beam_order
from evaluation.metrics import (
    METRIC_NAMES,
    compute_all,
    print_metrics,
    print_comparison,
)
from config import cfg


def reconstruct(fragment_samples):
    """Run the full pipeline directly without the LLM agent."""
    if fragment_samples and isinstance(fragment_samples[0], str):
        fragment_samples = [fragment_samples]

    fragments = fragment_samples[0]
    constraints = trypsin_filter(fragments)
    graph = build_overlap_graph(fragment_samples)
    scores = score_junctions(
        fragments,
        unscored_pairs=graph["unscored_junctions"],
        confirmed_junctions=graph["confirmed_junctions"],
    )
    order = beam_order(
        scores,
        impossible_junctions=constraints["impossible_junctions"],
        start_candidates=constraints["start_candidates"],
        confirmed_successors=graph["confirmed_successors"],
    )
    reconstruction = "".join(fragments[i] for i in order)
    return reconstruction, order, constraints, graph


test_path = cfg["data"]["ecoli_test_split"]
test_samples = cfg["data"].get("test_samples")

with open(test_path) as f:
    samples = [json.loads(l) for l in f if l.strip()][:test_samples]

baseline_summary = {k: [] for k in METRIC_NAMES}
recon_summary = {k: [] for k in METRIC_NAMES}
pruned_pcts = []

print(f"Sequential Evaluation ({len(samples)} Samples)")
print("-" * 60)

for i, sample in enumerate(samples, 1):
    target = sample.get("ecoli_original", sample.get("target_reconstruction"))
    fragment_samples = sample.get("fragment_samples") or [sample["fragments"]]
    fragments = fragment_samples[0]

    # Baseline: a fresh random permutation of the fragments.
    baseline_order = list(range(len(fragments)))
    random.Random(cfg["misc"]["seed"] + i).shuffle(baseline_order)
    baseline_recon = "".join(fragments[idx] for idx in baseline_order)
    baseline_metrics = compute_all(target, baseline_recon, fragments, baseline_order)

    # Reconstructed: pipeline output
    reconstruction, order, constraints, graph = reconstruct(fragment_samples)
    recon_metrics = compute_all(target, reconstruction, fragments, order)

    for k in METRIC_NAMES:
        baseline_summary[k].append(baseline_metrics[k])
        recon_summary[k].append(recon_metrics[k])

    n_frags = len(fragments)
    total_junctions = n_frags * (n_frags - 1)
    pruned = len(constraints["impossible_junctions"])
    pct = (pruned / total_junctions * 100) if total_junctions else 0.0
    pruned_pcts.append(pct)

    print(f"Sample {i}")
    print(f"  Target:         {target}")
    print(f"  Reconstruction: {reconstruction}")
    print(f"  Filter: {pruned}/{total_junctions} junctions pruned ({pct:.1f}%)")
    print(
        f"  Graph: {len(graph['confirmed_adjacencies'])} confirmed adjacencies, {len(graph['unscored_junctions'])} pairs pending scoring"
    )
    print_metrics(recon_metrics)

if samples:
    n = len(samples)
    print(f"\nAverage Results ({n} Samples) — Shuffled vs Reconstructed")
    print("-" * 60)
    print(f"  Model: {cfg['mlm_model']['name']}")
    print(f"  Beam Size: {cfg['mlm_model'].get('beam_size', 'N/A')}")
    print(f"  Junction Window: {cfg['mlm_model'].get('junction_window', 'N/A')}")
    print(f"  Missed Cleavage Ratio: {cfg['data'].get('missed_cleavage_ratio', 'N/A')}")
    avg_pruned = sum(pruned_pcts) / len(pruned_pcts)
    print(f"  Avg Junctions Pruned: {avg_pruned:.1f}%")
    print()
    print_comparison(baseline_summary, recon_summary, n)
