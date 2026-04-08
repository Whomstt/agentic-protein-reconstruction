import json
from algorithms.trypsin_filter import trypsin_filter
from algorithms.score_junctions import score_junctions
from algorithms.beam_order import beam_order
from evaluation.metrics import METRIC_NAMES, compute_all, print_metrics, print_averages
from config import cfg


def reconstruct(fragments):
    """Run the full pipeline directly without the LLM agent."""
    constraints = trypsin_filter(fragments)
    scores = score_junctions(fragments)
    order = beam_order(
        scores,
        start_candidates=constraints["start_candidates"],
        terminal_candidates=constraints["terminal_candidates"],
    )
    reconstruction = "".join(fragments[i] for i in order)
    return reconstruction, order


test_path = cfg["data"]["ecoli_test_split"]
sample_count = 10  # set to None to evaluate all samples

with open(test_path) as f:
    samples = [json.loads(l) for l in f if l.strip()][:sample_count]

summary = {k: [] for k in METRIC_NAMES}

print(f"Sequential Evaluation ({len(samples)} Samples)")
print("-" * 60)

for i, sample in enumerate(samples, 1):
    target = sample["ecoli_original"]

    fragments = sample["fragments"]

    reconstruction, order = reconstruct(fragments)
    metrics = compute_all(target, reconstruction, fragments, order)

    for k in METRIC_NAMES:
        summary[k].append(metrics[k])

    print(f"Sample {i}")
    print(f"  Target: {target}")
    print(f"  Reconstruction: {reconstruction}")
    print_metrics(metrics)

if samples:
    n = len(samples)
    print(f"\nAverage Results ({n} Samples)")
    print("-" * 60)
    print(f"  Model: {cfg['mlm_model']['name']}")
    print(f"  Beam Size: {cfg['mlm_model'].get('beam_size', 'N/A')}")
    print(f"  Missed Cleavage Ratio: {cfg['data'].get('missed_cleavage_ratio', 'N/A')}")
    print(f"  Minimum Peptide Length: {cfg['data'].get('min_length', 'N/A')}")
    print_averages(summary, n)
