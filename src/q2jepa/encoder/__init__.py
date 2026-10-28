import torch
import torch.nn.functional as F
from torch import nn


def prepare(x: torch.Tensor, res: int) -> torch.Tensor:
    """Downsample encoder input only; data, readout and metrics stay 64x64."""
    return x if x.shape[-1] == res else F.interpolate(x, size=(res, res), mode="area")


def build_encoder(cfg) -> nn.Module:
    from q2jepa.encoder.cnn import CNNEncoder

    if cfg.name == "cnn":
        return CNNEncoder(cfg.res, cfg.d)
    raise ValueError(f"unknown encoder {cfg.name!r}")
