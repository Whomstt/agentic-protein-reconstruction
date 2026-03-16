# Import necessary libraries
import itertools
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from config import cfg


def format_sequence(seq, is_prot):
    return " ".join(list(seq)) if is_prot else seq


def score_junctions(fragments):
    """Score all pairwise fragment junctions using batched ProtBERT MLM."""
    model_type = cfg["mlm_model"]["type"]
    batch_size = cfg["mlm_model"]["batch_size"]
    max_length = cfg["mlm_model"]["max_length"]
    if model_type == "prot":
        from models.prot import mlm, tokeniser

        is_prot = True
    elif model_type == "esm":
        from models.esm import mlm, tokeniser

        is_prot = False

    n = len(fragments)
    pairs = list(itertools.permutations(range(n), 2))
    inputs, targets = [], []
    sep = " " if is_prot else ""
    mask = tokeniser.mask_token

    for i, j in pairs:
        a, b = fragments[i], fragments[j]
        inputs.append(
            format_sequence(a, is_prot)
            + sep
            + mask
            + sep
            + format_sequence(b[1:], is_prot)
        )
        targets.append(b[0])
        inputs.append(
            format_sequence(a[:-1], is_prot)
            + sep
            + mask
            + sep
            + format_sequence(b, is_prot)
        )
        targets.append(a[-1])

    all_scores = []
    for start in tqdm(range(0, len(inputs), batch_size), desc="Scoring Junctions"):
        end = min(start + batch_size, len(inputs))
        batch_inputs = tokeniser(
            inputs[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batch_inputs = {k: v.to(cfg["device"]) for k, v in batch_inputs.items()}
        with torch.no_grad():
            logits = mlm(**batch_inputs).logits
        for k in range(end - start):
            # each input has exactly one mask
            mask_idx = (
                batch_inputs["input_ids"][k] == tokeniser.mask_token_id
            ).nonzero(as_tuple=False)[0, 0]
            score = F.log_softmax(logits[k, mask_idx], dim=-1)[
                tokeniser.convert_tokens_to_ids(targets[start + k])
            ].item()
            all_scores.append(score)

    mat = torch.zeros(n, n)
    for idx, (i, j) in enumerate(pairs):
        mat[i, j] = (all_scores[idx * 2] + all_scores[idx * 2 + 1]) / 2

    return mat
