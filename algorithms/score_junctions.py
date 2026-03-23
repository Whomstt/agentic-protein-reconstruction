# Import necessary libraries
import itertools
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from config import cfg


def format_sequence(seq, is_prot):
    return " ".join(list(seq)) if is_prot else seq


def score_junctions(fragments):
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
        batch_inputs = {k: v.to(cfg["misc"]["device"]) for k, v in batch_inputs.items()}
        with torch.no_grad():
            logits = mlm(**batch_inputs).logits
        for k in range(end - start):
            mask_idx = (
                batch_inputs["input_ids"][k] == tokeniser.mask_token_id
            ).nonzero(as_tuple=False)[0, 0]
            target_id = tokeniser.convert_tokens_to_ids(targets[start + k])
            score = F.log_softmax(logits[k, mask_idx], dim=-1)[target_id].item()

            frag_len = len(fragments[pairs[start + k][1]])
            all_scores.append(score / max(frag_len, 1))

    mat = torch.zeros(n, n)
    for idx, (i, j) in enumerate(pairs):
        mat[i, j] = all_scores[idx]

    return mat
