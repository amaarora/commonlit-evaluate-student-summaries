import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataset import CommonLitDataset
from model import CommonLitModel
import logging, wandb

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def train_one_epoch(model, optim, train_loader, device, scheduler=None):
    model.train()
    train_loss = []
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        content = batch["content"].to(device)
        wording = batch["wording"].to(device)
        output = model(input_ids, attention_mask)
        loss = (F.mse_loss(output, content) + F.mse_loss(output, wording)) / 2
        loss.backward()
        optim.step()
        if scheduler is not None:
            scheduler.step()
        train_loss.append(loss.item())
        wandb.log({"learning_rate": optim.param_groups[0]["lr"], "loss": loss.item()})
    return np.mean(train_loss)


if __name__ == "__main__":
    run = wandb.init(project="commonlit-evaluate-student-summaries")
    df = pd.read_csv("../data/train.csv")
    model_name = "distilbert-base-uncased"
    for prompt_id in df.prompt_id.unique():
        train_df = df[df.prompt_id != prompt_id]
        val_df = df[df.prompt_id == prompt_id]
        train_dataset = CommonLitDataset(model_name=model_name, df=train_df)
        val_dataset = CommonLitDataset(
            model_name=model_name, df=train_df, tok=train_dataset.tok
        )
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)
        model = CommonLitModel(model_name=model_name, tok_len=len(train_dataset.tok))
        opt = AdamW(model.parameters(), lr=1e-4)
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=10, num_training_steps=len(train_loader) * 1
        )
        train_loss = train_one_epoch(
            model, opt, train_loader=train_loader, device="cuda", scheduler=sched
        )