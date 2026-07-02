from threading import Lock

from transformers import EsmForMaskedLM, EsmTokenizer

from config import cfg

model_lock = Lock()


def load_model():
    tokeniser = EsmTokenizer.from_pretrained(cfg["validity_model"]["name"])
    mlm = EsmForMaskedLM.from_pretrained(
        cfg["validity_model"]["name"], low_cpu_mem_usage=False
    ).to(cfg["misc"]["device"])
    mlm.eval()
    return mlm, tokeniser


mlm, tokeniser = load_model()


def reset_cache(model=mlm):
    for layer in model.esm.encoder.layer:
        rotary = layer.attention.self.rotary_embeddings
        rotary._seq_len_cached = None
