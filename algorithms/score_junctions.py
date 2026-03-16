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
    if cfg["mlm_model"]["type"] == "prot":
        from models.prot import mlm, tokeniser

        is_prot = True
    elif cfg["mlm_model"]["type"] == "esm":
        from models.esm import mlm, tokeniser

        is_prot = False
    n = len(fragments)
    pairs = list(itertools.permutations(range(n), 2))  # all ordered pairs of fragments
    inputs = []  # sequences with a [MASK] at the junction
    targets = []  # first residue of the next fragment
    sep = " " if is_prot else ""
    for i, j in pairs:  # for each fragment pair
        a, b = fragments[i], fragments[j]
        targets.append(
            b[0]
        )  # we want to predict the first residue of the other fragment
        inputs.append(
            format_sequence(a, is_prot)
            + sep
            + tokeniser.mask_token
            + sep
            + format_sequence(b[1:], is_prot)
        )
    # fragment i + [MASK] + fragment j (without first residue)
    mat = torch.zeros(n, n)  # score matrix
    for start in tqdm(
        range(0, len(pairs), cfg["mlm_model"]["batch_size"]), desc="Scoring Junctions"
    ):  # process in batches and show progress
        end = min(
            start + cfg["mlm_model"]["batch_size"], len(pairs)
        )  # end index for batch
        batch_inputs = tokeniser(
            inputs[start:end], return_tensors="pt", padding=True, truncation=True
        )  # tokenise batch
        batch_inputs = {
            k: v.to(cfg["device"]) for k, v in batch_inputs.items()
        }  # move to gpu if available
        with torch.no_grad():
            logits = mlm(**batch_inputs).logits  # final linear layer logits
        for k, (i, j) in enumerate(pairs[start:end]):  # for each fragment pair in batch
            mask_index = (
                batch_inputs["input_ids"][k] == tokeniser.mask_token_id
            ).nonzero(as_tuple=False)[
                0, 0
            ]  # find the mask index
            probs = F.softmax(
                logits[k, mask_index], dim=-1
            )  # get the probability distribution at the mask position
            mat[i, j] = probs[
                tokeniser.convert_tokens_to_ids(targets[start + k])
            ].item()  # score for the correct next residue
    return mat
