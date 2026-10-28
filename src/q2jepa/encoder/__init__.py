import torch
import torch.nn.functional as F


def prepare(x: torch.Tensor, res: int) -> torch.Tensor:
    """Downsample encoder input only; data, readout and metrics stay 64x64."""
    return x if x.shape[-1] == res else F.interpolate(x, size=(res, res), mode="area")
