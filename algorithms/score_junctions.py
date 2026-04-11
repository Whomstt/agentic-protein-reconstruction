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
    window = cfg["mlm_model"].get("junction_window", 3)

    if model_type == "prot":
        from models.prot import mlm, tokeniser

        is_prot = True
    elif model_type == "esm":
        from models.esm import mlm, tokeniser

        is_prot = False

    n = len(fragments)
    pairs = list(itertools.permutations(range(n), 2))
    mask = tokeniser.mask_token
    sep = " " if is_prot else ""

    # Each (i, j) pair contributes up to `window` masked inputs — one per
    # position in the first residues of fragment j. Aggregating log-probs over
    # a window gives a much stronger junction signal than masking a single
    # residue, without requiring multiple simultaneous masks per forward pass.
    inputs, targets, entry_pair = [], [], []
    for i, j in pairs:
        a, b = fragments[i], fragments[j]
        w = min(window, len(b))
        for pos in range(w):
            left_b = b[:pos]
            right_b = b[pos + 1 :]
            left_str = format_sequence(a + left_b, is_prot)
            right_str = format_sequence(right_b, is_prot)
            parts = [p for p in (left_str, mask, right_str) if p]
            inputs.append(sep.join(parts))
            targets.append(b[pos])
            entry_pair.append((i, j))

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
            mask_positions = (
                batch_inputs["input_ids"][k] == tokeniser.mask_token_id
            ).nonzero(as_tuple=False)
            if len(mask_positions) == 0:
                # Truncated past the mask — treat as no-information.
                all_scores.append(0.0)
                continue
            mask_idx = mask_positions[0, 0]
            target_id = tokeniser.convert_tokens_to_ids(targets[start + k])
            score = F.log_softmax(logits[k, mask_idx], dim=-1)[target_id].item()
            all_scores.append(score)

    mat = torch.zeros(n, n)
    counts = torch.zeros(n, n)
    for idx, (i, j) in enumerate(entry_pair):
        mat[i, j] += all_scores[idx]
        counts[i, j] += 1
    counts = counts.clamp(min=1)
    mat = mat / counts  # average log-prob per masked position

    return mat
