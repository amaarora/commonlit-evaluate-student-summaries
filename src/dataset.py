"""`train.csv` has been created by running `python src/preprocess.py`"""
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import Any


class CommonLitDataset(Dataset):
    "Dataset for CommonLit Readability Prize, reading summary and text from csv file"

    def __init__(self, model, df) -> None:
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.add_tokens(
            [
                "<PROMPT>",
                "<SUMMARY>",
                "<PROMPT_TITLE>",
                "</PROMPT>",
                "</SUMMARY>",
                "</PROMPT_TITLE>",
            ]
        )
        self.df = df

    def __getitem__(self, idx) -> Any:
        text = self.df.iloc[idx].excerpt
        text = self.tok(
            text, padding="max_length", truncation=True, return_tensors="pt"
        )
        return (
            {
                "input_ids": text.input_ids.squeeze(),
                "attention_mask": text.attention_mask.squeeze(),
                "content": self.df.iloc[idx].content,
                "wording": self.df.iloc[idx].wording,
            }
            if "content" in self.df.columns
            else {
                "input_ids": text.input_ids.squeeze(),
                "attention_mask": text.attention_mask.squeeze(),
                "content": None,
                "wording": None,
            }
        )
