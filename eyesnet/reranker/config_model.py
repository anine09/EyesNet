from pydantic import BaseModel
from typing import List

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