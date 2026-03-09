# Import necessary libraries
import itertools
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from models.prot_bert import mlm, tokeniser, device


def score_junctions(fragments, batch_size=32):
    """Score all pairwise fragment junctions using batched ProtBERT MLM."""
    n = len(fragments)
    pairs = list(itertools.permutations(range(n), 2))  # all ordered pairs of fragments
    inputs = []  # sequences with a [MASK] at the junction
    targets = []  # first residue of the next fragment
    for i, j in pairs:  # for each fragment pair
        a, b = fragments[i], fragments[j]
        targets.append(
            b[0]
        )  # we want to predict the first residue of the other fragment
        inputs.append(
            " ".join(list(a)) + " [MASK] " + " ".join(list(b[1:]))
        )  # fragment i + [MASK] + fragment j (without first residue)
    mat = torch.zeros(n, n)  # score matrix
    for start in tqdm(
        range(0, len(pairs), batch_size), desc="Scoring Junctions"
    ):  # process in batches and show progress
        end = min(start + batch_size, len(pairs))  # end index for batch
        batch_inputs = tokeniser(
            inputs[start:end], return_tensors="pt", padding=True, truncation=True
        )  # tokenise batch
        batch_inputs = {
            k: v.to(device) for k, v in batch_inputs.items()
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
