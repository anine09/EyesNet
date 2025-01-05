import tomllib
from typing import List

import torch
import torch.optim as optim
from eyesnet_clip import EyesNetCLIP
from eyesnet_dataset import EyesNetDataset
from loguru import logger
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import autocast, GradScaler

# Set random seeds for reproducibility
import random

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

logger.add("train_clip.log", mode="w", level="INFO")


class GeneralConfig(BaseModel):
    dataset_path: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    warmup_epochs: int
    weight_decay: float
    patience: int
    cuda_device: int


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

dataset = EyesNetDataset(config.general.dataset_path)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
train_dataloader = DataLoader(
    train_dataset, batch_size=config.general.batch_size, shuffle=True, pin_memory=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=config.general.batch_size, pin_memory=True
)

device = torch.device(
    f"cuda:{config.general.cuda_device}" if torch.cuda.is_available() else "cpu"
)
model = EyesNetCLIP(config).to(device)

# Ensure that parameters requiring weight_decay=0 are handled correctly
optimizer = optim.AdamW(
    [
        {
            "params": model.xrd_encoder.parameters(),
            "weight_decay": config.general.weight_decay,
        },
        {
            "params": model.crystal_encoder.parameters(),
            "weight_decay": config.general.weight_decay,
        },
        {"params": model.logit_scale, "weight_decay": 0.0},
    ],
    lr=config.general.learning_rate,
)

warmup_epochs = config.general.warmup_epochs
warmup_scheduler = LinearLR(
    optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
)
main_scheduler = CosineAnnealingLR(
    optimizer, T_max=config.general.num_epochs - warmup_epochs, eta_min=1e-6
)
scheduler = SequentialLR(
    optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
)

best_crystal_acc = 0.0
best_xrd_acc = 0.0
early_stop_counter = 0

# Initialize GradScaler for mixed precision training
scaler = GradScaler("cuda")

for epoch in tqdm(range(config.general.num_epochs), desc="Epoch Progress", position=0):
    model.train()
    train_loss = 0.0

    batch_progress = tqdm(
        train_dataloader,
        desc=f"Batch Progress (Epoch {epoch+1})",
        leave=False,
        position=1,
    )

    for xrd, crystal in batch_progress:
        optimizer.zero_grad()
        xrd = xrd.to(device)
        crystal = crystal.to(device)

        # Enable autocasting for mixed precision
        with autocast("cuda"):
            loss, _, _ = model(xrd=xrd, crystal=crystal)

        if torch.isnan(loss):
            logger.error(f"Epoch {epoch+1}\tLoss is nan, skipping batch.")
            continue

        # Scaled backward pass and optimization
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Unscales gradients and calls optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        if any(
            torch.isnan(param.grad).any()
            for param in model.parameters()
            if param.grad is not None
        ):
            logger.error(f"Epoch {epoch+1}\tGradient is nan, skipping batch.")
            optimizer.zero_grad()
        else:
            optimizer.step()

        batch_progress.set_postfix(loss=loss.item())
        train_loss += loss.item()

    scheduler.step()

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

            with autocast():
                loss, crystal_acc, xrd_acc = model(xrd=xrd, crystal=crystal)

            if torch.isnan(loss):
                logger.error(
                    f"Epoch {epoch+1}\tValidation loss is nan, skipping batch."
                )
                continue

            val_loss += loss.item()
            crystal_test_accs.append(crystal_acc)
            xrd_test_accs.append(xrd_acc)

    avg_val_loss = val_loss / len(test_dataloader)
    avg_crystal_acc = sum(crystal_test_accs) / len(crystal_test_accs)
    avg_xrd_acc = sum(xrd_test_accs) / len(xrd_test_accs)

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
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= config.general.patience:
            logger.warning(f"Early stopping triggered after {epoch+1} epochs.")
            break
