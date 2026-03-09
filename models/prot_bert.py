# Import necessary libraries
from transformers import BertForMaskedLM, BertTokenizer
from config import DEVICE


# Load the pretrained ProtBERT MLM model
tokeniser = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
mlm = BertForMaskedLM.from_pretrained("Rostlab/prot_bert").to(DEVICE)
mlm.eval()  # evaluation mode
