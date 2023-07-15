"""`train.csv` has been created by running `python src/preprocess.py`"""
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import Any
import torch


class CommonLitDataset(Dataset):
    "Dataset for CommonLit Readability Prize, reading summary and text from csv file"

    def __init__(self, model_name, df, tok=None) -> None:
        super().__init__()
        if not tok:
            self.tok = AutoTokenizer.from_pretrained(model_name)
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
        else:
            self.tok = tok
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Any:
        text = self.df.iloc[idx].excerpt
        text = self.tok(
            text, padding="max_length", truncation=True, return_tensors="pt"
        )
        return (
            {
                "input_ids": text.input_ids.squeeze(),
                "attention_mask": text.attention_mask.squeeze(),
                "content": torch.tensor(self.df.iloc[idx].content, dtype=torch.float),
                "wording": torch.tensor(self.df.iloc[idx].wording, dtype=torch.float),
            }
            if "content" in self.df.columns
            else {
                "input_ids": text.input_ids.squeeze(),
                "attention_mask": text.attention_mask.squeeze(),
                "content": None,
                "wording": None,
            }
        )
