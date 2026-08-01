"""Derived analyses over a completed run's stored per-sample data.

Pure functions plus one loader: no model, no GPU, no network. Everything comes
from ``samples.jsonl`` and the config snapshot in ``summary.json``, so the report
layer re-runs offline over runs that finished months ago.

Two facts about the stored data drive the design, both checked in
``tests/test_analysis.py``:

1. ``fragments = fragment_samples[0]`` and the digest is emitted left-to-right,
   so replica 0 tiles the target in order and the ground-truth order is the
   identity permutation in each run's fragment index space.
2. Fragment strings are not stored and duplicates are common, so index-space
   adjacency counting disagrees with the string-multiset metrics the run
   reported on ~10-30% of samples. Anything adjacency-shaped is therefore
   derived from the stored metric values, never recounted from ``order``.

The exception is the N-terminal check (``order[0] == 0``), exact because the
N-terminal peptide is never duplicated; validated on 197/197 samples whose
fragment strings survive."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.metrics import (
    METRIC_NAMES,
    edit_similarity,
    nanmean,
    rank_concordance,
)

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

METRIC_KEYS = tuple(METRIC_NAMES)

# Arm name -> the per-sample key holding that arm's metrics. Order is the
# method ladder used by every table and figure.
ARMS = {
    "shuffled": "baseline_metrics",
    "deterministic": "first_pass_metrics",
    "control": "control_metrics",
    "agentic": "recon_metrics",
    "oracle": "oracle_metrics",
}

ARM_LABELS = {
    "shuffled": "Random Order",
    "deterministic": "Fixed Settings",
    "control": "Random Search (no LLM)",
    "agentic": "LLM-Guided",
    "oracle": "Best Candidate (ceiling)",
}

# Fragment-count bins. A protein's difficulty scales with how many pieces it was
# cut into, so every headline number is also reported stratified by this.
FRAGMENT_BINS = ((2, 4), (5, 9), (10, 19), (20, 49), (50, None))

# --- Error taxonomy thresholds -------------------------------------------
# DISCLOSED CHOICES. These cut points are not tuned against any outcome; they
# are round numbers chosen to separate qualitatively different failure shapes,
# and every one of them is applied to STORED, string-correct metric values.
# Changing them changes only how failures are labelled, never any headline
# metric. They are printed in the report so a reader can re-derive the split.
TAXONOMY_THRESHOLDS = {
    "reversal_tau_max": -0.5,  # strongly anti-correlated ordering
    "scramble_tau_abs_max": 0.15,  # no global ordering signal
    "scramble_apa_max": 0.05,  # essentially no correct adjacencies
    "local_max_breakpoints": 2,  # at most two joins wrong
    "local_min_lcr": 0.5,  # ...while half the protein is one correct block
}

TAXONOMY_LABELS = {
    "exact": "Exact reconstruction",
    "block_reversal": "Block reversal",
    "local_transposition": "Local transposition",
    "wrong_start": "Wrong start (structured, misanchored)",
    "partial_assembly": "Partial assembly (correct start)",
    "full_scramble": "Full scramble",
    "unknown": "Unclassified (no ground-truth order)",
}

TAXONOMY_ORDER = (
    "exact",
    "local_transposition",
    "wrong_start",
    "partial_assembly",
    "block_reversal",
    "full_scramble",
    "unknown",
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class Run:
    """One completed run: its config snapshot and per-sample records."""

    path: Path
    name: str
    config: dict
    samples: list[dict]
    summary: dict = field(default_factory=dict)

    @property
    def organism(self) -> str:
        data = self.config.get("data", {})
        return data.get("organism_display_name") or data.get("organism") or "unknown"

    @property
    def replica_count(self):
        return self.config.get("data", {}).get("replica_count")

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def has_control(self) -> bool:
        return any("control_metrics" in s for s in self.samples)

    @property
    def has_oracle(self) -> bool:
        return any("oracle_metrics" in s for s in self.samples)

    @property
    def arms(self) -> list[str]:
        """Arms actually produced by this run, in ladder order."""
        present = []
        for arm, key in ARMS.items():
            if any(key in s for s in self.samples):
                present.append(arm)
        return present

    def label(self) -> str:
        rc = self.replica_count
        return f"{self.organism} r{rc}" if rc is not None else self.organism


def _backfill_edit_similarity(sample: dict) -> None:
    """Derive ``edit_similarity`` for runs finished before the metric existed.

    Those runs stored every arm's reconstruction *string*, so the metric is a
    direct recomputation rather than an estimate — no re-running, and a run that
    already stores the key is left untouched.

    The shuffled arm is the one exception: it stores ``baseline_order`` (an index
    permutation) but not the string or the fragment list it indexes, and the
    fragment lists for the r20/r100 runs no longer exist. It stays NaN, which
    ``nanmean`` drops, rather than being reconstructed approximately.
    """
    target = sample.get("target")
    if not isinstance(target, str) or not target:
        return

    def put(metrics, reconstruction):
        if not isinstance(metrics, dict) or "edit_similarity" in metrics:
            return
        metrics["edit_similarity"] = (
            edit_similarity(target, reconstruction)
            if isinstance(reconstruction, str) and reconstruction
            else float("nan")
        )

    for key in ("iteration_history", "control_iteration_history"):
        for record in sample.get(key) or []:
            if isinstance(record, dict):
                put(record.get("metrics"), record.get("reconstruction"))

    history = sample.get("iteration_history") or []
    control_history = sample.get("control_iteration_history") or []

    put(sample.get("recon_metrics"), sample.get("reconstruction"))
    put(sample.get("first_pass_metrics"), (history[0] or {}).get("reconstruction") if history else None)

    best = sample.get("control_best_iteration")
    control_best = None
    if isinstance(best, int) and 1 <= best <= len(control_history):
        control_best = (control_history[best - 1] or {}).get("reconstruction")
    put(sample.get("control_metrics"), control_best)

    # Shuffled floor: runner.py keeps its index permutation but discards the
    # string it built them from, so the sequence has to come back from the
    # fragment list. sample_diagnostics stores that list, but only for runs made
    # after it was added — older ones stay NaN, since the digestion RNG is
    # unseeded and the fragment set cannot be regenerated.
    fragments = sample.get("fragments")
    baseline_order = sample.get("baseline_order")
    baseline_recon = None
    if fragments and baseline_order and len(baseline_order) == len(fragments):
        if sorted(baseline_order) == list(range(len(fragments))):
            baseline_recon = "".join(fragments[i] for i in baseline_order)
    put(sample.get("baseline_metrics"), baseline_recon)

    # Oracle mirrors runner.py: the best value over the candidates the agent
    # actually generated, so it is a max over the same per-iteration records.
    oracle = sample.get("oracle_metrics")
    if isinstance(oracle, dict) and "edit_similarity" not in oracle:
        finite = [
            r["metrics"]["edit_similarity"]
            for r in history
            if isinstance(r, dict)
            and isinstance(r.get("metrics", {}).get("edit_similarity"), (int, float))
            and not math.isnan(r["metrics"]["edit_similarity"])
        ]
        oracle["edit_similarity"] = max(finite) if finite else float("nan")


def load_run(run_dir) -> Run:
    """Read one results folder. ``samples.jsonl`` is the source of truth for
    per-sample data; ``summary.json`` is read only for the config snapshot and
    run name (its ``samples`` list duplicates the jsonl)."""
    path = Path(run_dir)
    samples_path = path / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"{samples_path} not found — not a run folder")

    samples = []
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sample = json.loads(line)
                _backfill_edit_similarity(sample)
                samples.append(sample)

    config: dict = {}
    name = path.name
    summary: dict = {}
    summary_path = path / "summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        config = summary.get("config", {}) or {}
        name = summary.get("run_name") or name
        summary.pop("samples", None)  # large and redundant with samples.jsonl

    return Run(path=path, name=name, config=config, samples=samples, summary=summary)


def discover_runs(root=RESULTS_ROOT) -> list[Path]:
    """All run folders under ``results/`` that contain a samples.jsonl, oldest
    first by folder name."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "samples.jsonl").exists())


# --------------------------------------------------------------------------
# Per-sample derivations
# --------------------------------------------------------------------------


def fragment_count(sample: dict) -> int:
    """Number of fragments the protein was digested into."""
    order = sample.get("order") or sample.get("baseline_order") or []
    if order:
        return len(order)
    total = sample.get("total_junctions") or 0
    if total > 0:  # total = n(n-1) -> solve for n
        return int(round((1 + math.sqrt(1 + 4 * total)) / 2))
    return 0


def fragment_bin(n: int) -> str:
    """Bin label for a fragment count, e.g. '10-19' or '50+'."""
    for low, high in FRAGMENT_BINS:
        if high is None:
            if n >= low:
                return f"{low}+"
        elif low <= n <= high:
            return f"{low}-{high}"
    return "<2" if n < 2 else "?"


def bin_labels() -> list[str]:
    return [f"{lo}+" if hi is None else f"{lo}-{hi}" for lo, hi in FRAGMENT_BINS]


def _metric(sample: dict, arm: str, metric: str):
    block = sample.get(ARMS[arm]) or {}
    value = block.get(metric)
    return value if isinstance(value, (int, float)) else None


def n_terminal_correct(order) -> bool | None:
    """Did the ordering place the true first fragment first?

    True order is the identity permutation, so this is ``order[0] == 0``. Unlike
    adjacency counting the index proxy is exact here: the N-terminal peptide is
    never a duplicate of another fragment."""
    if not order:
        return None
    return order[0] == 0


def breakpoints(sample: dict, arm: str = "agentic"):
    """Wrong joins in the ordering: ``(n-1) - correct_adjacencies``.

    Derived from the stored adjacent-pair accuracy, which was computed on fragment
    strings as a multiset; an index-space recount disagrees with it wherever
    duplicate fragments exist (28/99 and 9/98 samples on the two checkable runs).
    None when the ordering metrics are NaN (fragments did not tile)."""
    n = fragment_count(sample)
    apa = _metric(sample, arm, "adjacent_pair_acc")
    if n < 2 or apa is None or math.isnan(apa):
        return None
    return (n - 1) * (1.0 - apa)


def classify_error(sample: dict, arm: str = "agentic") -> str:
    """Label the failure shape from stored, string-correct metric values.

    Categories are checked most specific first, so each sample lands in exactly one.
    A reversal or a scramble is a shape of failure that subsumes a wrong start, so
    those are tested before the start check and only an ordering with real structure
    that begins on the wrong fragment is labelled ``wrong_start``. Cut points are in
    TAXONOMY_THRESHOLDS."""
    em = _metric(sample, arm, "exact_match")
    apa = _metric(sample, arm, "adjacent_pair_acc")
    lcr = _metric(sample, arm, "longest_correct_run")
    tau = _metric(sample, arm, "kendall_tau")
    t = TAXONOMY_THRESHOLDS

    if em is not None and em >= 1.0:
        return "exact"
    if any(v is None or math.isnan(v) for v in (apa, lcr, tau)):
        return "unknown"

    bp = breakpoints(sample, arm)
    start_ok = n_terminal_correct(sample.get("order") if arm == "agentic" else None)

    if tau <= t["reversal_tau_max"]:
        return "block_reversal"
    if abs(tau) < t["scramble_tau_abs_max"] and apa <= t["scramble_apa_max"]:
        return "full_scramble"
    if bp is not None and bp <= t["local_max_breakpoints"] and lcr >= t["local_min_lcr"]:
        return "local_transposition"
    if start_ok is False:
        return "wrong_start"
    return "partial_assembly"


def sample_concordance(sample: dict, quality_metric: str = "adjacent_pair_acc", control=False):
    """Within-sample concordance between the validity signal and true quality across
    the iterations this sample tried; ~0.50 is chance.

    The run keeps the lowest-validity candidate, so if validity does not track true
    quality the Oracle-Agentic gap cannot be closed by better search alone."""
    key = "control_iteration_history" if control else "iteration_history"
    history = sample.get(key) or []
    pairs = []
    for record in history:
        validity = record.get("validity_score")
        quality = (record.get("metrics") or {}).get(quality_metric)
        if isinstance(validity, (int, float)) and isinstance(quality, (int, float)):
            pairs.append((validity, quality))
    return rank_concordance(pairs)


def sample_row(run: Run, sample: dict) -> dict:
    """One flat, CSV-ready record per sample: every raw per-sample metric and
    field, plus the derived quantities. This is the single source the CSV
    export, the figures and every table read from."""
    n = fragment_count(sample)
    order = sample.get("order") or []
    row: dict = {
        "run_dir": run.path.name,
        "run_name": run.name,
        "organism": run.organism,
        "replica_count": run.replica_count,
        "sample_index": sample.get("index"),
        "protein_length": len(sample.get("target") or ""),
        "num_fragments": n,
        "fragment_bin": fragment_bin(n),
        "true_order_recovered": (sample.get("recon_metrics") or {}).get(
            "true_order_recovered"
        ),
        "completed": sample.get("completed"),
    }

    for arm in run.arms:
        for metric in METRIC_KEYS:
            row[f"{arm}_{metric}"] = _metric(sample, arm, metric)

    row["nterm_correct"] = n_terminal_correct(order)
    row["nterm_correct_shuffled"] = n_terminal_correct(sample.get("baseline_order") or [])
    row["breakpoints"] = breakpoints(sample, "agentic")
    row["breakpoints_normalized"] = (
        row["breakpoints"] / (n - 1) if row["breakpoints"] is not None and n > 1 else None
    )
    row["error_class"] = classify_error(sample, "agentic")

    concordance, comparable = sample_concordance(sample)
    row["validity_concordance"] = concordance
    row["validity_comparable_pairs"] = comparable

    row.update(
        {
            "best_iteration": sample.get("best_iteration"),
            "num_iterations": sample.get("num_iterations"),
            "best_validity_score": sample.get("best_validity_score"),
            "first_pass_validity_score": sample.get("first_pass_validity_score"),
            "control_validity_score": sample.get("control_validity_score"),
            "control_best_iteration": sample.get("control_best_iteration"),
            "llm_calls": sample.get("llm_calls"),
            "llm_tokens": sample.get("llm_tokens"),
            "llm_failures": sample.get("llm_failures"),
            "duration_seconds": sample.get("duration_seconds"),
            "agentic_duration_seconds": sample.get("agentic_duration_seconds"),
            "control_duration_seconds": sample.get("control_duration_seconds"),
            "num_pruned": sample.get("num_pruned"),
            "total_junctions": sample.get("total_junctions"),
            "pruned_pct": sample.get("pruned_pct"),
            "num_confirmed_adjacencies": (sample.get("graph") or {}).get(
                "num_confirmed_adjacencies"
            ),
        }
    )

    # Optional fields, written by runs from the instrumentation change onward.
    # Absent on older runs, which is why every consumer treats them as missing
    # rather than assuming a value.
    for optional in ("junction_ranking", "trypsin_recall", "true_order", "fragments"):
        if optional in sample:
            value = sample[optional]
            if optional == "junction_ranking" and isinstance(value, dict):
                for k in ("top1_acc", "top3_acc", "mrr", "num_junctions"):
                    row[f"junction_{k}"] = value.get(k)
            elif optional == "trypsin_recall" and isinstance(value, dict):
                for k, v in value.items():
                    row[f"trypsin_{k}"] = v
    return row


def sample_rows(run: Run) -> list[dict]:
    return [sample_row(run, s) for s in run.samples]


# --------------------------------------------------------------------------
# Aggregations
# --------------------------------------------------------------------------


def arm_values(rows: list[dict], arm: str, metric: str) -> list:
    """Per-sample values for one arm/metric, in sample order (pairing intact)."""
    return [r.get(f"{arm}_{metric}") for r in rows]


def exact_match_count(rows: list[dict], arm: str) -> tuple[int, int]:
    """(successes, n) for Exact Match — a binomial count, not a mean, so it can
    go through a Wilson interval instead of a bootstrap."""
    values = [v for v in arm_values(rows, arm, "exact_match") if isinstance(v, (int, float))]
    return sum(1 for v in values if v >= 1.0), len(values)


def stratify_by_bin(rows: list[dict], arm: str, metric: str) -> dict:
    """Group per-sample values into fragment-count bins.

    Returns bin label -> {'values': [...], 'n': int}. Empty bins are retained so
    a table or figure shows the gap rather than silently closing it.
    """
    out = {label: {"values": [], "n": 0} for label in bin_labels()}
    for row in rows:
        label = row.get("fragment_bin")
        if label not in out:
            continue
        value = row.get(f"{arm}_{metric}")
        if isinstance(value, (int, float)) and not math.isnan(value):
            out[label]["values"].append(value)
        out[label]["n"] += 1
    return out


def lift_over_baseline(rows: list[dict], metric: str, arm: str = "agentic") -> dict:
    """Per fragment-count bin: the arm's mean, the shuffled floor, and the lift
    (paired per-sample difference, so it is not a difference of two independent
    means). Lift is the number that says whether the method is doing anything
    at that difficulty."""
    out = {}
    for label in bin_labels():
        paired = [
            (r.get(f"{arm}_{metric}"), r.get(f"shuffled_{metric}"))
            for r in rows
            if r.get("fragment_bin") == label
        ]
        usable = [
            (a, b)
            for a, b in paired
            if isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and not math.isnan(a)
            and not math.isnan(b)
        ]
        out[label] = {
            "n": len(paired),
            "n_usable": len(usable),
            "arm_mean": nanmean([a for a, _ in usable]),
            "baseline_mean": nanmean([b for _, b in usable]),
            "lift": nanmean([a - b for a, b in usable]),
            "deltas": [a - b for a, b in usable],
        }
    return out


def taxonomy_counts(rows: list[dict]) -> dict:
    counts = {key: 0 for key in TAXONOMY_ORDER}
    for row in rows:
        key = row.get("error_class") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def nterm_analysis(rows: list[dict]) -> dict:
    """P(correct N-terminal start) and Exact Match conditioned on it.

    The start fragment anchors a left-to-right greedy assembly, so this
    separates "never got going" from "started right and drifted".
    """
    known = [r for r in rows if isinstance(r.get("nterm_correct"), bool)]
    correct = [r for r in known if r["nterm_correct"]]
    wrong = [r for r in known if not r["nterm_correct"]]

    def em_rate(subset):
        values = [
            r.get("agentic_exact_match")
            for r in subset
            if isinstance(r.get("agentic_exact_match"), (int, float))
        ]
        if not values:
            return None, 0, 0
        successes = sum(1 for v in values if v >= 1.0)
        return successes / len(values), successes, len(values)

    rate_correct, hits_correct, n_correct = em_rate(correct)
    rate_wrong, hits_wrong, n_wrong = em_rate(wrong)
    shuffled_known = [r for r in rows if isinstance(r.get("nterm_correct_shuffled"), bool)]
    return {
        "n": len(known),
        "n_correct_start": len(correct),
        "p_correct_start": len(correct) / len(known) if known else None,
        "shuffled_p_correct_start": (
            sum(1 for r in shuffled_known if r["nterm_correct_shuffled"]) / len(shuffled_known)
            if shuffled_known
            else None
        ),
        "em_given_correct_start": rate_correct,
        "em_hits_given_correct_start": hits_correct,
        "n_given_correct_start": n_correct,
        "em_given_wrong_start": rate_wrong,
        "em_hits_given_wrong_start": hits_wrong,
        "n_given_wrong_start": n_wrong,
    }


def breakpoint_stats(rows: list[dict]) -> dict:
    values = [r["breakpoints"] for r in rows if isinstance(r.get("breakpoints"), (int, float))]
    normalized = [
        r["breakpoints_normalized"]
        for r in rows
        if isinstance(r.get("breakpoints_normalized"), (int, float))
    ]
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": ordered[len(ordered) // 2],
        "min": ordered[0],
        "max": ordered[-1],
        "mean_normalized": nanmean(normalized),
        "zero_breakpoint_samples": sum(1 for v in values if v < 0.5),
    }


def concordance_summary(rows: list[dict]) -> dict:
    values = [
        r["validity_concordance"]
        for r in rows
        if isinstance(r.get("validity_concordance"), (int, float))
    ]
    pairs = sum(r.get("validity_comparable_pairs") or 0 for r in rows)
    return {
        "n_samples": len(values),
        "mean_concordance": nanmean(values),
        "comparable_pairs": pairs,
        "above_chance": sum(1 for v in values if v > 0.5),
    }


def oracle_gap(rows: list[dict]) -> dict:
    """Per metric: how much true quality the imperfect validity selection left
    on the table, among candidates the run had already generated."""
    out = {}
    for metric in METRIC_KEYS:
        deltas = [
            r[f"oracle_{metric}"] - r[f"agentic_{metric}"]
            for r in rows
            if isinstance(r.get(f"oracle_{metric}"), (int, float))
            and isinstance(r.get(f"agentic_{metric}"), (int, float))
            and not math.isnan(r[f"oracle_{metric}"])
            and not math.isnan(r[f"agentic_{metric}"])
        ]
        out[metric] = {
            "mean_gap": nanmean(deltas),
            "n": len(deltas),
            "samples_with_gap": sum(1 for d in deltas if d > 1e-12),
        }
    return out


def cost_summary(rows: list[dict]) -> dict:
    def total(key):
        return sum(r.get(key) or 0 for r in rows)

    n = len(rows) or 1
    return {
        "n_samples": len(rows),
        "total_llm_calls": total("llm_calls"),
        "total_llm_tokens": total("llm_tokens"),
        "llm_calls_per_sample": total("llm_calls") / n,
        "llm_tokens_per_sample": total("llm_tokens") / n,
        "llm_failures": total("llm_failures"),
        "seconds_per_sample": total("duration_seconds") / n,
        "agentic_seconds_per_sample": total("agentic_duration_seconds") / n,
        "control_seconds_per_sample": total("control_duration_seconds") / n,
        "completed": sum(1 for r in rows if r.get("completed")),
        "true_order_recovered": sum(1 for r in rows if r.get("true_order_recovered")),
    }


def junction_ranking_summary(rows: list[dict]) -> dict | None:
    """Aggregate the per-sample junction-ranking diagnostic when the run stored
    it. Returns None for runs written before that instrumentation existed, so
    the report prints an explicit 'n/a - requires field' line rather than a
    fabricated number."""
    have = [r for r in rows if isinstance(r.get("junction_top1_acc"), (int, float))]
    if not have:
        return None
    return {
        "n_samples": len(have),
        "top1_acc": nanmean([r["junction_top1_acc"] for r in have]),
        "top3_acc": nanmean([r["junction_top3_acc"] for r in have]),
        "mrr": nanmean([r["junction_mrr"] for r in have]),
        "total_junctions": sum(r.get("junction_num_junctions") or 0 for r in have),
    }


def trypsin_recall_summary(rows: list[dict]) -> dict | None:
    """Aggregate trypsin-filter recall (did pruning ever remove a TRUE junction?)
    when the run stored the pruned mask. None on older runs."""
    have = [r for r in rows if isinstance(r.get("trypsin_true_junctions_pruned"), (int, float))]
    if not have:
        return None
    pruned = sum(r["trypsin_true_junctions_pruned"] for r in have)
    total = sum(r.get("trypsin_true_junctions") or 0 for r in have)
    return {
        "n_samples": len(have),
        "true_junctions_pruned": pruned,
        "true_junctions": total,
        "recall": (total - pruned) / total if total else None,
        "samples_with_loss": sum(1 for r in have if r["trypsin_true_junctions_pruned"] > 0),
    }
