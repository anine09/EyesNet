import torch
import torch.nn as nn
import torch.nn.functional as F
from crystal_encoder.gotennet import GotenNet
from torch_geometric.nn import global_mean_pool
from xrd_encoder.ByteLatentTransformer import ByteLatentTransformer
from icecream import ic



def CLIP_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate a custom cross-entropy loss.

    Args:
    - logits (torch.Tensor): The input tensor containing unnormalized logits.

    Returns:
    - torch.Tensor: The computed custom cross-entropy loss.

    Example:
    >>> logits = torch.rand((batch_size, num_classes))
    >>> loss = CLIP_loss(logits)
    """
    n = logits.shape[1]
    # Create labels tensor
    labels = torch.arange(n)
    # bring logits to cpu
    logits = logits.to("cpu")
    # Calculate cross entropy losses along axis 0 and 1
    loss_crystal = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
    loss_xrd = F.cross_entropy(logits, labels, reduction="mean")
    # Calculate the final loss
    loss = (loss_crystal + loss_xrd) / 2
    return loss


def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    crystal2xrd_match_idx = similarity.argmax(dim=1)
    xrd2crystal_match_idx = similarity.argmax(dim=0)

    crystal_acc = (crystal2xrd_match_idx == y).float().mean()
    xrd_acc = (xrd2crystal_match_idx == y).float().mean()
    return crystal_acc, xrd_acc


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class XRDEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.base = ByteLatentTransformer(**cfg.xrd_encoder.dict())
        self.projection = Projection(
            cfg.xrd_encoder.n_embd, cfg.projection.hidden_dim, cfg.projection.dropout
        )

    def forward(self, x):
        out = self.base(x)
        out = out.mean(dim=1)
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class CrystalEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.base = GotenNet(**cfg.crystal_encoder.dict())
        self.projection = Projection(
            cfg.crystal_encoder.n_atom_basis, cfg.projection.hidden_dim, cfg.projection.dropout
        )

    def forward(self, crystal):
        out, _ = self.base(crystal)
        out = global_mean_pool(out, crystal.batch)
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class EyesNetCLIP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.xrd_encoder = XRDEncoder(cfg)
        self.crystal_encoder = CrystalEncoder(cfg)
        self.lr = cfg.general.learning_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, xrd, crystal):
        xrd_embed = self.xrd_encoder(xrd)
        crystal_embed = self.crystal_encoder(crystal)
        similarity = xrd_embed @ crystal_embed.T
        loss = CLIP_loss(similarity)
        
        crystal_acc, xrd_acc = metrics(similarity)
        return loss, crystal_acc, xrd_acc
