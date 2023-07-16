import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataset import CommonLitDataset
from model import CommonLitModel
import logging, wandb
from early_stopping import EarlyStopping
from average_meter import AverageMeter

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def train_one_epoch(model, optim, train_loader, device, scheduler=None):
    loss_meter = AverageMeter()
    model.train()
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        content = batch["content"].to(device)
        wording = batch["wording"].to(device)
        output = model(input_ids, attention_mask)
        loss = (
            F.mse_loss(output[:, 0], content) + F.mse_loss(output[:, 1], wording)
        ) / 2
        loss.backward()
        optim.step()
        if scheduler is not None:
            scheduler.step()
        loss_meter.update(loss.item(), len(batch))
        wandb.log({"learning_rate": optim.param_groups[0]["lr"]})
    return loss_meter.avg


def validate_one_epoch(model, val_loader, device):
    "Validate one epoch and also calculate mean columnwise root mean squared error"
    loss_meter = AverageMeter()
    model.eval()
    content_preds = []
    wording_preds = []
    content_targets = []
    wording_targets = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            content = batch["content"].to(device)
            wording = batch["wording"].to(device)
            output = model(input_ids, attention_mask)
            content_preds.append(output[:, 0].cpu().numpy())
            wording_preds.append(output[:, 1].cpu().numpy())
            content_targets.append(content.cpu().numpy())
            wording_targets.append(wording.cpu().numpy())
            loss = (
                F.mse_loss(output[:, 0], content) + F.mse_loss(output[:, 1], wording)
            ) / 2
            loss_meter.update(loss.item(), len(batch))
        # calculate mean columnwise root mean squared error for content and wording
        content_preds = np.concatenate(content_preds)
        wording_preds = np.concatenate(wording_preds)
        content_targets = np.concatenate(content_targets)
        wording_targets = np.concatenate(wording_targets)
        logging.info(
            f"Content preds shae: {content_preds.shape}, Wording preds shape: {wording_preds.shape}, Content targets shape: {content_targets.shape}, Wording targets shape: {wording_targets.shape}"
        )
        content_rmse = np.sqrt(np.mean((content_preds - content_targets) ** 2, axis=0))
        wording_rmse = np.sqrt(np.mean((wording_preds - wording_targets) ** 2, axis=0))
        mcrmse = np.mean([content_rmse, wording_rmse])
    return loss_meter.avg, mcrmse


if __name__ == "__main__":
    run = wandb.init(project="commonlit-evaluate-student-summaries")
    df = pd.read_csv("../data/train.csv")
    model_name = "distilbert-base-uncased"
    N_EPOCHS = 3
    for prompt_id in df.prompt_id.unique():
        logging.info(f"Validation prompt ID: {prompt_id}")
        train_df = df[df.prompt_id != prompt_id]
        val_df = df[df.prompt_id == prompt_id]
        logging.info(f"Train df: {train_df.shape}, Val df: {val_df.shape}")
        train_dataset = CommonLitDataset(model_name=model_name, df=train_df)
        val_dataset = CommonLitDataset(
            model_name=model_name, df=train_df, tok=train_dataset.tok
        )
        logging.info(f"Created train & test dataset")

        train_loader = DataLoader(train_dataset, batch_size=96)
        val_loader = DataLoader(val_dataset, batch_size=96)

        logging.info("Creating model")
        model = CommonLitModel(model_name=model_name, tok_len=len(train_dataset.tok))
        model.to("cuda")
        opt = AdamW(model.parameters(), lr=1e-4)
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=10, num_training_steps=len(train_loader) * N_EPOCHS
        )
        es = EarlyStopping(total_epochs=N_EPOCHS, patience=2, mode="min")
        for epoch in range(N_EPOCHS):
            logging.info(f"Epoch: {epoch}")
            train_loss = train_one_epoch(
                model, opt, train_loader=train_loader, device="cuda", scheduler=sched
            )
            val_loss, mcrmse = validate_one_epoch(
                model, val_loader=val_loader, device="cuda"
            )
            wandb.log(
                {
                    "train_epoch_loss": train_loss,
                    "val_epoch_loss": val_loss,
                    "mc_rmse": mcrmse,
                    "epoch": epoch,
                }
            )
