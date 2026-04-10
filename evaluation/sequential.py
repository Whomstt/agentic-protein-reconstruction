import json
from algorithms.trypsin_filter import trypsin_filter
from algorithms.score_junctions import score_junctions
from algorithms.beam_order import beam_order
from evaluation.metrics import (
    METRIC_NAMES,
    compute_all,
    print_metrics,
    print_comparison,
)
from config import cfg


def reconstruct(fragments):
    """Run the full pipeline directly without the LLM agent."""
    constraints = trypsin_filter(fragments)
    scores = score_junctions(fragments)
    order = beam_order(
        scores,
        impossible_junctions=constraints["impossible_junctions"],
        start_candidates=constraints["start_candidates"],
    )
    reconstruction = "".join(fragments[i] for i in order)
    return reconstruction, order, constraints


test_path = cfg["data"]["ecoli_test_split"]
sample_count = 100  # set to None to evaluate all samples

with open(test_path) as f:
    samples = [json.loads(l) for l in f if l.strip()][:sample_count]

baseline_summary = {k: [] for k in METRIC_NAMES}
recon_summary = {k: [] for k in METRIC_NAMES}
pruned_pcts = []

print(f"Sequential Evaluation ({len(samples)} Samples)")
print("-" * 60)

for i, sample in enumerate(samples, 1):
    target = sample["ecoli_original"]
    fragments = sample["fragments"]

    # Baseline: fragments in their (shuffled) input order
    baseline_order = list(range(len(fragments)))
    baseline_recon = "".join(fragments)
    baseline_metrics = compute_all(target, baseline_recon, fragments, baseline_order)

    # Reconstructed: pipeline output
    reconstruction, order, constraints = reconstruct(fragments)
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
    print_metrics(recon_metrics)

if samples:
    n = len(samples)
    print(f"\nAverage Results ({n} Samples) — Shuffled vs Reconstructed")
    print("-" * 60)
    print(f"  Model: {cfg['mlm_model']['name']}")
    print(f"  Beam Size: {cfg['mlm_model'].get('beam_size', 'N/A')}")
    print(f"  Missed Cleavage Ratio: {cfg['data'].get('missed_cleavage_ratio', 'N/A')}")
    print(f"  Minimum Peptide Length: {cfg['data'].get('min_length', 'N/A')}")
    avg_pruned = sum(pruned_pcts) / len(pruned_pcts)
    print(f"  Avg Junctions Pruned: {avg_pruned:.1f}%")
    print()
    print_comparison(baseline_summary, recon_summary, n)
