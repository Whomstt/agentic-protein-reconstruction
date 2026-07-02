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

    masked_inputs = input_ids.repeat(sequence_length, 1)
    masked_positions = torch.arange(sequence_length, device=masked_inputs.device) + 1
    masked_inputs[
        torch.arange(sequence_length, device=masked_inputs.device), masked_positions
    ] = tokeniser.mask_token_id
    repeated_attention = attention_mask.repeat(sequence_length, 1)

    log_probs = []
    with model_lock:
        for start in range(0, sequence_length, batch_size):
            end = min(start + batch_size, sequence_length)
            batch_inputs = masked_inputs[start:end]
            batch_attention = repeated_attention[start:end]
            batch_positions = masked_positions[start:end]
            batch_targets = input_ids[0, batch_positions]

            with torch.no_grad():
                logits = mlm(
                    input_ids=batch_inputs, attention_mask=batch_attention
                ).logits

            rows = torch.arange(end - start, device=logits.device)
            position_logits = logits[rows, batch_positions]
            scores = F.log_softmax(position_logits, dim=-1)[rows, batch_targets]
            log_probs.append(scores)

    mean_log_prob = torch.cat(log_probs).mean().item()
    return float(math.exp(-mean_log_prob))
