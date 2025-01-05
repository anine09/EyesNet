import math
import tomllib
from typing import List

import torch
import torch.optim as optim
from eyesnet_clip import EyesNetCLIP
from eyesnet_dataset import EyesNetDataset
from loguru import logger
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Set random seeds for reproducibility
import random

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

logger.remove()
logger.add("train_clip.log", mode="w", level="INFO")


class GeneralConfig(BaseModel):
    dataset_path: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    warmup_epochs: int
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
    train_dataset, batch_size=config.general.batch_size, shuffle=True
)
test_dataloader = DataLoader(test_dataset, batch_size=config.general.batch_size)

device = torch.device(
    f"cuda:{config.general.cuda_device}" if torch.cuda.is_available() else "cpu"
)
model = EyesNetCLIP(config).to(device)

optimizer = optim.Adam(
    [
        {"params": model.xrd_encoder.parameters()},
        {"params": model.crystal_encoder.parameters()},
    ],
    lr=config.general.learning_rate,
)

def lr_lambda(current_epoch):
    if current_epoch < config.general.warmup_epochs:
        return float(current_epoch) / float(max(1, config.general.warmup_epochs))
    else:
        return 0.5 * (
            1
            + math.cos(
                math.pi
                * (current_epoch - config.general.warmup_epochs)
                / (config.general.num_epochs - config.general.warmup_epochs)
            )
        )

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = GradScaler()

best_val_loss = float("inf")
early_stop_counter = 0

for epoch in tqdm(range(config.general.num_epochs), desc="Epoch Progress", position=0):
    model.train()
    train_loss = 0.0

    for xrd, crystal in tqdm(
        train_dataloader,
        desc=f"Batch Progress (Epoch {epoch+1})",
        leave=False,
        position=1,
    ):
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=True):
            xrd = xrd.to(device)
            crystal = crystal.to(device)
            loss, _, _ = model(xrd=xrd, crystal=crystal)
            train_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    avg_train_loss = train_loss / len(train_dataloader)
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch {epoch+1}\tLearning Rate: {current_lr}\tAverage Training Loss: {avg_train_loss}")

    model.eval()
    val_loss = 0.0
    crystal_test_accs = []
    xrd_test_accs = []

    with torch.no_grad():
        for xrd, crystal in test_dataloader:
            xrd = xrd.to(device)
            crystal = crystal.to(device)
            loss, crystal_acc, xrd_acc = model(xrd=xrd, crystal=crystal)
            val_loss += loss.item()
            crystal_test_accs.append(crystal_acc)
            xrd_test_accs.append(xrd_acc)

    avg_val_loss = val_loss / len(test_dataloader)
    avg_crystal_acc = sum(crystal_test_accs) / len(crystal_test_accs)
    avg_xrd_acc = sum(xrd_test_accs) / len(xrd_test_accs)

    logger.success(
        f"Epoch {epoch+1}\tValidation Loss: {avg_val_loss}\tCrystal Acc: {avg_crystal_acc}\tXRD Acc: {avg_xrd_acc}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, 'best_model.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= config.general.patience:
            logger.warning(f"Early stopping triggered after {epoch+1} epochs.")
            break