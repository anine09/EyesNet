
import tomllib
from typing import List

import torch
import torch.optim as optim
from eyesnet_clip import EyesNetCLIP
from eyesnet_dataset import EyesNetDataset
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from icecream import ic
from torch.amp import autocast, GradScaler



class GeneralConfig(BaseModel):
    dataset_path: str
    learning_rate: float
    padding_len: int
    batch_size: int
    num_epochs: int
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
    dim: int
    max_degree: int
    depth: int
    heads: int
    dim_head: int
    dim_edge_refinement: int
    return_coors: bool
    num_atoms: int
    cutoff_radius: float


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

dataset = EyesNetDataset(
    config.general.dataset_path, padding=config.general.padding_len
)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
train_dataloder = DataLoader(
    train_dataset, batch_size=config.general.batch_size, shuffle=True
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = EyesNetCLIP(config).to(device)
optimizer = optim.Adam(
    [
        {"params": model.xrd_encoder.parameters()},
        {"params": model.crystal_encoder.parameters()},
    ],
    lr=model.lr,
)


scaler = GradScaler("cuda")
for epoch in range(config.general.num_epochs):
    model.train()
    for data in train_dataloder:
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=True):
            xrd, atom_ids, coordinates, adjacency_matrix = data
            xrd = torch.round(xrd).int()
            xrd = xrd.to(device)
            atom_ids = atom_ids.to(device)
            coordinates = coordinates.to(device)
            adjacency_matrix = adjacency_matrix.to(device)
            loss, crystal_acc, xrd_acc = model(
                xrd=xrd,
                atom_ids=atom_ids,
                coordinates=coordinates,
                adjacency_matrix=adjacency_matrix,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(epoch, loss.item(), crystal_acc, xrd_acc)