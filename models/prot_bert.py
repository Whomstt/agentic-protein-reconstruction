# Import necessary libraries
import torch
from transformers import BertForMaskedLM, BertTokenizer


# Load the pretrained ProtBERT MLM model
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # set device to cpu or gpu if available
print(f"Using device: {device}")
tokeniser = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
mlm = BertForMaskedLM.from_pretrained("Rostlab/prot_bert").to(device)
mlm.eval()  # evaluation mode
