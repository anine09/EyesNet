import os
import random
import tomllib
from typing import List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from eyesnet_clip import EyesNetCLIP
from eyesnet_dataset import EyesNetDataset
from loguru import logger
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

logger.add("train_clip.log", mode="a", level="INFO")


class GeneralConfig(BaseModel):
    dataset_path: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    warmup_steps: int
    lr_decay: float
    patience: int
    nan_patience: int
    gradient_clipping: float


class XRDEncoderConfig(BaseModel):
    block_size: int
    patch_size: int
    n_embd: int
    n_head: int
    n_layers_encoder: int
    n_layers_latent: int
    dropout: float
    vocab_size: int
    hash_sizes: List[int]
    hash_table_size: int


class CrystalEncoderConfig(BaseModel):
    n_atom_basis: int
    n_interactions: int
    radial_basis: str
    n_rbf: int
    cutoff: float
    max_z: int
    epsilon: float
    max_num_neighbors: int
    num_heads: int
    attn_dropout: float
    edge_updates: bool
    scale_edge: bool
    lmax: int
    aggr: str
    sep_int_vec: bool


class ProjectionConfig(BaseModel):
    hidden_dim: int
    dropout: float


class Config(BaseModel):
    general: GeneralConfig
    xrd_encoder: XRDEncoderConfig
    crystal_encoder: CrystalEncoderConfig
    projection: ProjectionConfig


with open("config.toml", "rb") as f:
    config = tomllib.load(f)
config = Config(**config)

dataset = EyesNetDataset(config.general.dataset_path, ICSD=False)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.01)
train_dataloader = DataLoader(
    train_dataset, batch_size=config.general.batch_size, shuffle=True, pin_memory=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=config.general.batch_size, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyesNetCLIP(config).to(device)

optimizer = optim.AdamW(
    [
        {
            "params": model.xrd_encoder.parameters(),
        },
        {
            "params": model.crystal_encoder.parameters(),
        },
    ],
    lr=config.general.learning_rate,
    weight_decay=config.general.weight_decay,
)


# 线性预热函数
def warmup_lr_scheduler(step):
    if step < config.general.warmup_steps:
        return float(step) / float(max(1, config.general.warmup_steps))
    return 1.0


warmup_scheduler = LambdaLR(optimizer, warmup_lr_scheduler)
reduce_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.general.lr_decay, patience=config.general.patience)


if os.path.exists("latest_step_model.pth.pth"):
    checkpoint = torch.load("latest_step_model.pth", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_crystal_acc = checkpoint["best_crystal_acc"]
    best_xrd_acc = checkpoint["best_xrd_acc"]
    logger.info(f"Resuming training from epoch {start_epoch}")
else:
    best_crystal_acc = 0.0
    best_xrd_acc = 0.0
    start_epoch = 0
    logger.info("Starting training from scratch.")


latest_model_save_flag = True
early_stop_counter = 0
nan_epoch_counter = 0


for epoch in tqdm(
    range(start_epoch, config.general.num_epochs), desc="Epoch Progress", position=0
):
    model.train()
    train_loss = 0.0

    batch_progress = tqdm(
        train_dataloader,
        desc=f"Batch Progress (Epoch {epoch+1})",
        leave=False,
        position=1,
    )

    for idx, (xrd, crystal) in enumerate(batch_progress):
        optimizer.zero_grad()
        xrd = xrd.to(device)
        crystal = crystal.to(device)
        loss, _, _ = model(xrd=xrd, crystal=crystal)

        if torch.isnan(loss):
            logger.error(f"Epoch {epoch+1}\tLoss is nan, skipping batch.")
            continue

        # Scaled backward pass and optimization
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.general.gradient_clipping
        )

        if any(
            torch.isnan(param.grad).any()
            for param in model.parameters()
            if param.grad is not None
        ):
            logger.error(f"Epoch {epoch+1}\tGradient is nan, skipping batch.")
            optimizer.zero_grad()
        else:
            optimizer.step()

        optimizer.step()
        warmup_scheduler.step()  # 更新预热学习率

        batch_progress.set_postfix(loss=f"{loss.item():.2f}")
        train_loss += loss.item()

        if idx % 1000 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_crystal_acc": best_crystal_acc,
                    "best_xrd_acc": best_xrd_acc,
                },
                "latest_step_model.pth",
            )

    avg_train_loss = train_loss / len(train_dataloader)
    current_lr = optimizer.param_groups[0]["lr"]
    logger.info(
        f"Epoch {epoch+1}\tLearning Rate: {current_lr}\tAverage Training Loss: {avg_train_loss}"
    )

    model.eval()
    val_loss = 0.0
    crystal_test_accs = []
    xrd_test_accs = []

    with torch.no_grad():
        for xrd, crystal in test_dataloader:
            xrd = xrd.to(device)
            crystal = crystal.to(device)
            loss, crystal_acc, xrd_acc = model(xrd=xrd, crystal=crystal)

            if torch.isnan(loss):
                logger.error(
                    f"Epoch {epoch+1}\tValidation loss is nan, skipping batch."
                )
                latest_model_save_flag = False
                nan_epoch_counter += 1
                break
            else:
                nan_epoch_counter = 0

            val_loss += loss.item()
            crystal_test_accs.append(crystal_acc)
            xrd_test_accs.append(xrd_acc)

    avg_val_loss = val_loss / len(test_dataloader)
    avg_crystal_acc = sum(crystal_test_accs) / len(crystal_test_accs)
    avg_xrd_acc = sum(xrd_test_accs) / len(xrd_test_accs)

    reduce_scheduler.step(avg_crystal_acc + avg_xrd_acc)


    torch.cuda.empty_cache()

    logger.success(
        f"Epoch {epoch+1}\tValidation Loss: {avg_val_loss}\tCrystal Acc: {avg_crystal_acc}\tXRD Acc: {avg_xrd_acc}"
    )

    if (avg_crystal_acc + avg_xrd_acc) > (best_crystal_acc + best_xrd_acc):
        best_crystal_acc = max(avg_crystal_acc, best_crystal_acc)
        best_xrd_acc = max(avg_xrd_acc, best_xrd_acc)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_crystal_acc": best_crystal_acc,
                "best_xrd_acc": best_xrd_acc,
            },
            "best_model.pth",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_crystal_acc": best_crystal_acc,
                "best_xrd_acc": best_xrd_acc,
            },
            "latest_model.pth",
        )
        logger.success(
            f"Best Crystal Acc: {best_crystal_acc}\tBest XRD Acc: {best_xrd_acc}\tEpoch: {epoch+1}"
        )
        early_stop_counter = 0
    else:
        # early_stop_counter += 1
        # logger.info(
        #     f"Early stopping counter: {early_stop_counter}/{config.general.patience}"
        # )
        if latest_model_save_flag:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_crystal_acc": best_crystal_acc,
                    "best_xrd_acc": best_xrd_acc,
                },
                "latest_model.pth",
            )
            logger.info(f"Latest model saved at epoch {epoch+1}")
        elif nan_epoch_counter >= config.general.nan_patience:
            logger.error("Stop Training because a lot of NaN.")
            raise RuntimeError("Stop Training because a lot of NaN.")
        # if early_stop_counter >= config.general.patience:
        #     logger.warning(f"Early stopping triggered after {epoch+1} epochs.")
        #     break
