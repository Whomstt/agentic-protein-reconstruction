import itertools
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from config import cfg
from models.memory import free_gpu_memory


def format_sequence(seq, is_prot):
    return " ".join(list(seq)) if is_prot else seq


def score_junctions(
    fragments,
    unscored_pairs=None,
    confirmed_junctions=None,
    window=None,
    junction_pairs=None,
    existing_scores=None,
):
    model_type = cfg["mlm_model"]["type"]
    batch_size = cfg["mlm_model"]["batch_size"]
    max_length = cfg["mlm_model"]["max_length"]
    window = (
        cfg["search"]["default_levers"]["junction_window"]
        if window is None
        else int(window)
    )

    def _normalize_pairs(pairs):
        if not pairs:
            return set()
        normalized = set()
        for pair in pairs:
            if len(pair) != 2:
                continue
            i, j = pair
            normalized.add((int(i), int(j)))
        return normalized

    if model_type == "prot":
        from models.prot import mlm, model_lock, reset_cache, tokeniser

        reset_cache(mlm)
        is_prot = True
    elif model_type == "esm":
        from models.esm import mlm, model_lock, reset_cache, tokeniser

        reset_cache(mlm)
        is_prot = False

    n = len(fragments)
    all_pairs = list(itertools.permutations(range(n), 2))
    if junction_pairs is not None and existing_scores is not None:
        pair_filter = _normalize_pairs(junction_pairs)
        pairs = [pair for pair in all_pairs if pair in pair_filter]
    elif unscored_pairs is not None:
        pairs = [pair for pair in all_pairs if pair in set(unscored_pairs)]
    else:
        pairs = all_pairs
    mask = tokeniser.mask_token
    sep = " " if is_prot else ""

    # Each (i, j) pair contributes up to `window` masked inputs, one per
    # position in the first residues of fragment j; log-probs are aggregated
    # over the window below.
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
    with model_lock:
        for batch_idx, start in enumerate(
            tqdm(range(0, len(inputs), batch_size), desc="Scoring Junctions")
        ):
            end = min(start + batch_size, len(inputs))
            batch_inputs = tokeniser(
                inputs[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            batch_inputs = {
                k: v.to(cfg["misc"]["device"]) for k, v in batch_inputs.items()
            }
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
            del batch_inputs, logits
            # Periodically release cached GPU memory; a high replica_count can
            # push this loop into thousands of batches.
            if batch_idx % 50 == 0:
                free_gpu_memory()

    mat = existing_scores.clone() if existing_scores is not None else torch.zeros(n, n)
    pair_scores = {}
    pair_counts = {}
    for idx, (i, j) in enumerate(entry_pair):
        pair_scores[(i, j)] = pair_scores.get((i, j), 0.0) + all_scores[idx]
        pair_counts[(i, j)] = pair_counts.get((i, j), 0) + 1

    for (i, j), score_sum in pair_scores.items():
        mat[i, j] = score_sum / pair_counts[(i, j)]

    if confirmed_junctions:
        for i, j in confirmed_junctions:
            mat[i, j] = 1_000.0

    return mat
