import math

import torch
import torch.nn.functional as F

from config import cfg


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
