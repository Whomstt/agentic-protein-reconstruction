# Import necessary libraries
from transformers import EsmForMaskedLM, EsmTokenizer
from config import cfg


# Load the pretrained ESM-2 MLM model
tokeniser = EsmTokenizer.from_pretrained(cfg["mlm_model"]["name"])
mlm = EsmForMaskedLM.from_pretrained(cfg["mlm_model"]["name"]).to(cfg["device"])
mlm.eval()  # evaluation mode
