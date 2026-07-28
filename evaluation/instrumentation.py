"""Optional per-sample diagnostics recorded into samples.jsonl.

Instrumentation, not logic: every field is derived from state the pipeline has
already produced, after the reconstruction is finished. No model calls, no
change to any search input, so a run's reconstruction, metrics and cost are
identical whether or not this is called, and the caller swallows any exception
for the same reason.

Every field is optional. Older runs lack them and ``evaluation/analysis.py``
reports 'n/a - requires field: X' rather than assuming a value."""

from __future__ import annotations

from evaluation.metrics import junction_ranking_stats, recover_true_order


def sample_diagnostics(target: str, fragments, state_snapshot: dict) -> dict:
    """Diagnostics for one finished sample, as optional fields to merge into the
    sample record:

    ``fragments``       replica-0 fragment strings. Not otherwise stored, and the
                        digestion RNG is unseeded, so without this a run's fragment
                        set is lost once the dataset is regenerated.
    ``true_order``      ground-truth permutation, so no consumer re-derives it.
    ``junction_ranking``top-1/top-3/MRR of the true successor under the run's own
                        score matrix, over the junctions the pLM actually had to
                        discriminate (confirmed adjacencies excluded).
    ``trypsin_recall``  whether constraint pruning ever removed a true junction,
                        which the pruned count alone cannot answer."""
    out: dict = {}
    if not fragments:
        return out

    out["fragments"] = list(fragments)
    true_order = recover_true_order(target, fragments)
    out["true_order"] = true_order
    if true_order is None or len(true_order) < 2:
        return out

    n = len(fragments)
    true_junctions = [
        (true_order[k], true_order[k + 1]) for k in range(len(true_order) - 1)
    ]

    # --- junction scorer ranking -----------------------------------------
    scores = state_snapshot.get("scores")
    confirmed = state_snapshot.get("confirmed_junctions") or set()
    if scores:
        confirmed_pairs = {tuple(pair) for pair in confirmed}
        stats = junction_ranking_stats(scores, true_order, n, skip_pairs=confirmed_pairs)
        stats["excluded_confirmed"] = sum(
            1 for pair in true_junctions if pair in confirmed_pairs
        )
        stats["matrix"] = "run score matrix (confirmed junctions excluded)"
        out["junction_ranking"] = stats

    # --- trypsin filter recall -------------------------------------------
    impossible = state_snapshot.get("impossible_junctions")
    if impossible is not None:
        impossible_pairs = {tuple(pair) for pair in impossible}
        pruned_true = [pair for pair in true_junctions if pair in impossible_pairs]
        out["trypsin_recall"] = {
            "true_junctions": len(true_junctions),
            "true_junctions_pruned": len(pruned_true),
            "recall": (
                (len(true_junctions) - len(pruned_true)) / len(true_junctions)
                if true_junctions
                else None
            ),
            "pruned_total": len(impossible_pairs),
        }

    return out


def safe_sample_diagnostics(target: str, fragments, state_snapshot: dict) -> dict:
    """``sample_diagnostics`` that can never raise.

    A finished sample represents real GPU and LLM spend; a bug in an optional
    diagnostic must not be able to lose it.
    """
    try:
        return sample_diagnostics(target, fragments, state_snapshot or {})
    except Exception as exc:  # pragma: no cover - defensive by intent
        return {"diagnostics_error": f"{type(exc).__name__}: {exc}"}
