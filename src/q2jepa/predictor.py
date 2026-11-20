"""Code predictors P_psi(h, z), Q-JEPA Eqs. 5-6: shared (MovingMNIST, N = LayerNorm), agent-local (HEP18), scaled (D3IL)."""

import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, d: int = 64, n_bits: int = 3, width: int = 96):
        super().__init__()
        self.code = nn.Sequential(nn.Linear(n_bits, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.context = nn.Linear(d, width)
        self.transition = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        u = self.code(2 * z - 1)  # (B, K, width)
        c = self.context(h)[:, None].expand_as(u)
        return self.norm(h[:, None] + self.transition(torch.cat([c, u, c * u], dim=-1)))


class AgentLocalPredictor(nn.Module):
    """HEP18 (App. A.2, F.3): residual per-agent predictor with masked local codes and no output normalization.

    Each agent sees the full scene h and only its own bits (the others' signed bits are zeroed before the shared
    code MLP); the position of the unmasked bits tells the shared network which agent it serves. It outputs that
    agent's change in its own (d / agents)-dim slice of h.
    """

    def __init__(self, d: int = 64, n_bits: int = 12, width: int = 64, agents: int = 4):
        super().__init__()
        if d % agents or n_bits % agents:
            raise ValueError(f"d={d} and n_bits={n_bits} must split evenly over {agents} agents")
        self.agents, self.d_agent = agents, d // agents
        mask = torch.zeros(agents, n_bits)
        for a in range(agents):
            mask[a, a * (n_bits // agents) : (a + 1) * (n_bits // agents)] = 1
        self.register_buffer("mask", mask)
        self.code = nn.Sequential(nn.Linear(n_bits, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.context = nn.Linear(d, width)
        self.transition = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, self.d_agent))

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        b, k, _ = z.shape
        u = self.code((2 * z - 1)[:, :, None] * self.mask)  # (B, K, agents, width)
        c = self.context(h)[:, None, None].expand_as(u)
        delta = self.transition(torch.cat([c, u, c * u], dim=-1))  # (B, K, agents, d_agent)
        return (h.reshape(b, 1, self.agents, self.d_agent) + delta).reshape(b, k, -1)


class ScaledPredictor(Predictor):
    """D3IL (App. B.2, Eq. 15): the shared predictor plus a bounded scale/offset head on its hidden activation.

    t_hat = q + (s - 1)(q - b_LN) + w_LN * b with s = 2 sigmoid(a), b = 0.25 tanh(c); zero init gives s = 1, b = 0.
    """

    def __init__(self, d: int = 64, n_bits: int = 8, width: int = 128):
        super().__init__(d, n_bits, width)
        self.head = nn.Linear(width, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        u = self.code(2 * z - 1)
        c = self.context(h)[:, None].expand_as(u)
        hidden = self.transition[1](self.transition[0](torch.cat([c, u, c * u], dim=-1)))
        q = self.norm(h[:, None] + self.transition[2](hidden))
        a, off = self.head(hidden).unbind(-1)
        s, b = 2 * torch.sigmoid(a)[..., None], 0.25 * torch.tanh(off)[..., None]
        return q + (s - 1) * (q - self.norm.bias) + self.norm.weight * b


def build_predictor(cfg, d: int, n_bits: int, agents: int = 1) -> nn.Module:
    if cfg.kind == "shared":
        return Predictor(d, n_bits, cfg.width)
    if cfg.kind == "agent_local":
        return AgentLocalPredictor(d, n_bits, cfg.width, agents)
    if cfg.kind == "scaled":
        return ScaledPredictor(d, n_bits, cfg.width)
    raise ValueError(f"unknown predictor.kind {cfg.kind!r}")
