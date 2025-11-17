import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()  # loads .env into environment

class Tokenizer:
    def __init__(self) -> None:
        hf_token = os.getenv("HF_TOKEN")  # your HF token in .env
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            token=hf_token,
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def get(self):
        return self.tokenizer

