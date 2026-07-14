from threading import Lock

from transformers import BertForMaskedLM, BertTokenizer

from config import cfg

model_lock = Lock()


def load_model():
    tokeniser = BertTokenizer.from_pretrained(cfg["mlm_model"]["name"])
    mlm = BertForMaskedLM.from_pretrained(
        cfg["mlm_model"]["name"], low_cpu_mem_usage=False
    ).to(cfg["misc"]["device"])
    mlm.eval()
    return mlm, tokeniser


mlm, tokeniser = load_model()


def reset_cache(model=mlm):
    return None
