from transformers import AutoModel
import torch.nn as nn


class CommonLitModel(nn.Module):
    "CommonLitModel for CommonLit Readability Prize target prediction"

    def __init__(self, model_name, tok_len) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(tok_len)
        self.linear = nn.Linear(self.model.config.dim, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.linear(self.relu(output.last_hidden_state[:, 0, :]))
        return output
