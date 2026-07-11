import math

import torch
import torch.nn.functional as F

from config import cfg


def junction_local_logprob(fragments, order, window) -> float | None:
    """Mean masked-LM log-prob over ONLY the fragment junctions the ordering
    uses (successor's first `window` residues at each boundary), via the same
    junction-scoring model the pipeline already uses. Returns None when there
    is no junction to score. Higher (less negative) = more plausible.

    Whole-sequence pseudo-perplexity averages over every residue, but candidate
    orderings differ only at fragment junctions, so ~95% of that average is
    identical across candidates and drowns the ordering signal. Scoring only the
    junctions keeps just the residues whose likelihood actually depends on the
    order."""
    from algorithms.score_junctions import score_junctions

    if not order or len(order) < 2:
        return None
    pairs = [(order[k], order[k + 1]) for k in range(len(order) - 1)]
    # Score only these junctions with a fixed validity window so the signal is
    # comparable across iterations regardless of the window the agent searched
    # with. Don't pass confirmed_junctions (that would overwrite raw MLM scores);
    # confirmed edges are handled separately by confirmed_agreement().
    mat = score_junctions(fragments, unscored_pairs=pairs, window=window)
    vals = [mat[i][j].item() for i, j in pairs]
    return sum(vals) / len(vals) if vals else None


def confirmed_agreement(order, confirmed_junctions) -> float | None:
    """Fraction of the overlap graph's confirmed directed adjacencies that the
    ordering realizes as consecutive fragments. None when there are no confirmed
    edges. Higher = better; this is a near-ground-truth structural signal (real
    multi-replica overlaps, not model guesses) and strengthens with replica
    count."""
    if not confirmed_junctions or not order or len(order) < 2:
        return None
    realized = {(order[k], order[k + 1]) for k in range(len(order) - 1)}
    confirmed = {(int(i), int(j)) for i, j in confirmed_junctions}
    if not confirmed:
        return None
    return sum(1 for edge in confirmed if edge in realized) / len(confirmed)


def blended_validity(
    fragments,
    order,
    confirmed_junctions,
    junction_window,
    confirmed_penalty,
) -> float:
    """Selection signal for best-candidate choice (lower = better).

    = junction-local pseudo-perplexity, inflated when the ordering disagrees
    with overlap-confirmed adjacencies:

        j_ppl * (1 + confirmed_penalty * (1 - confirmed_agreement))

    Offline this reaches ~62% concordance with true quality on yeast (vs ~47%
    for whole-sequence pseudo-perplexity, which was worse than a coin flip) and
    holds ~57-60% on E. coli, so it is far more organism-robust as the basis for
    picking the best reconstruction."""
    mean_lp = junction_local_logprob(fragments, order, junction_window)
    if mean_lp is None:
        return float("inf")
    j_ppl = math.exp(-mean_lp)
    agreement = confirmed_agreement(order, confirmed_junctions)
    penalty = 0.0 if agreement is None else confirmed_penalty * (1.0 - agreement)
    score = j_ppl * (1.0 + penalty)
    if math.isnan(score):
        return float("inf")
    return float(score)


def pseudo_perplexity(sequence: str) -> float:
    if not sequence:
        return float("inf")

    from models.esm_validity import mlm, model_lock, reset_cache, tokeniser

    reset_cache(mlm)

    max_length = cfg["validity_model"]["max_length"]
    batch_size = cfg["validity_model"]["batch_size"]

    encoded = tokeniser(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(cfg["misc"]["device"])
    attention_mask = encoded["attention_mask"].to(cfg["misc"]["device"])

    sequence_length = input_ids.shape[1] - 2
    if sequence_length <= 0:
        return float("inf")

    # Build only the current batch's masked copies rather than materializing
    # an (L, L) matrix for the whole sequence up front — for long sequences
    # (a full mixture-organism target, or a long single protein) that matrix
    # was the single largest allocation in the run and it sat on the GPU for
    # the whole call even though only one batch_size-worth was ever in use at
    # a time.
    log_probs = []
    with model_lock:
        for start in range(0, sequence_length, batch_size):
            end = min(start + batch_size, sequence_length)
            batch_n = end - start
            rows = torch.arange(batch_n, device=input_ids.device)
            batch_positions = (
                torch.arange(start, end, device=input_ids.device) + 1
            )

            batch_inputs = input_ids.repeat(batch_n, 1)
            batch_inputs[rows, batch_positions] = tokeniser.mask_token_id
            batch_attention = attention_mask.repeat(batch_n, 1)
            batch_targets = input_ids[0, batch_positions]

            with torch.no_grad():
                logits = mlm(
                    input_ids=batch_inputs, attention_mask=batch_attention
                ).logits

            position_logits = logits[rows, batch_positions]
            scores = F.log_softmax(position_logits, dim=-1)[rows, batch_targets]
            log_probs.append(scores)
            del batch_inputs, batch_attention, logits, position_logits

    mean_log_prob = torch.cat(log_probs).mean().item()
    return float(math.exp(-mean_log_prob))
