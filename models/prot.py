# Import necessary libraries
from transformers import BertForMaskedLM, BertTokenizer
from config import cfg


# Load the pretrained ProtBERT MLM model
tokeniser = BertTokenizer.from_pretrained(cfg["mlm_model"]["name"])
mlm = BertForMaskedLM.from_pretrained(cfg["mlm_model"]["name"]).to(cfg["device"])
mlm.eval()  # evaluation mode
