import torch
from dotenv import load_dotenv
import os


load_dotenv()  # load environment variables from .env file

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # set device to cpu or gpu if available

# Models
PROTBERT_MODEL = "Rostlab/prot_bert"
LLM_MODEL = "gpt-5-nano"

# ProtBERT settings
FRAG_BATCH_SIZE = 32

# API Keys
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
