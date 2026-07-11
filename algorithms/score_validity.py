import math

import torch
import torch.nn.functional as F

from config import cfg


def junction_local_logprob(fragments, order, window, confirmed_junctions=None) -> float | None:
    """Mean masked-LM log-prob over the fragment junctions the ordering uses
    (successor's first `window` residues at each boundary), skipping any
    junction the overlap graph already confirmed by real multi-replica overlap
    evidence — those are already known-valid, so spending noisy PLM scoring on
    them would only dilute the signal; confirmed_agreement() scores them
    instead. Returns None when there is no junction left to score. Higher
    (less negative) = more plausible."""
    from algorithms.score_junctions import score_junctions

    if not order or len(order) < 2:
        return None
    confirmed = {(int(i), int(j)) for i, j in (confirmed_junctions or [])}
    pairs = [
        (order[k], order[k + 1])
        for k in range(len(order) - 1)
        if (order[k], order[k + 1]) not in confirmed
    ]
    if not pairs:
        return 0.0
    mat = score_junctions(fragments, unscored_pairs=pairs, window=window)
    vals = [mat[i][j].item() for i, j in pairs]
    return sum(vals) / len(vals)


def confirmed_agreement(order, confirmed_junctions) -> float | None:
    """Fraction of the overlap graph's confirmed directed adjacencies that the
    ordering realizes as consecutive fragments. None when there are no
    confirmed edges."""
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
    """Selection signal for best-candidate choice (lower = better):

        j_ppl * (1 + confirmed_penalty * (1 - confirmed_agreement))

    where j_ppl is junction-local pseudo-perplexity, inflated when the
    ordering disagrees with overlap-confirmed adjacencies."""
    mean_lp = junction_local_logprob(fragments, order, junction_window, confirmed_junctions)
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
