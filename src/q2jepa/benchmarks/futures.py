"""Shared benchmark for whole-scene clips with J equally likely futures (charged particles, chess).

A subclass provides `load(split)` -> compact observation and future sources (ball positions, board codes), shape
(n, *src) and (n, J, *src), and future states (n, J, m); `render(src)` -> clips (..., 3, 64, 64); `src_dims`
(trailing dims of one source); `scale`, the state unit; and optionally `perms`, state permutations that describe
the same scene. A subclass may also replace the readout head (`readout_dim`, `readout_fit_loss`, `decode`), e.g.
chess classifies squares instead of regressing them; metrics always see decoded states. Sources stay on the device and are rendered per batch, so the train split can be large (the
readout overfit 800 scenes, 2026-11-22). The encoder sees whole clips (dataset.agents = 1); targets are the J
encoded futures. The readout decodes [h; t] to the state; metrics match decoded samples to the J true futures
(metrics.futures_metrics).
"""

import numpy as np
import torch

from q2jepa.benchmarks.mmnist import encode_clips
from q2jepa.metrics import futures_metrics, match_futures
from q2jepa.readout.futures import StateReadout, readout_loss
from q2jepa.seeding import READOUT_SEED_OFFSET, reseed, save_atomic

ENCODE_CHUNK = 128  # clips per encoder call: bounds HQViT's CUDA-Q register, as in HEP18


class FuturesBenchmark:
    src_dims: int = 0
    scale: float = 1.0
    perms: list[list[int]] | None = None
    max_radius: float = float("inf")  # cap on the match radius, in units of scale (metrics.match_futures)

    def load(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def render(self, src: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __init__(self, cfg, device: torch.device):
        if cfg.dataset.agents != 1:
            raise ValueError(f"{cfg.dataset.name} encodes whole scenes: dataset.agents must be 1")
        self.cfg, self.device = cfg, device
        self.obs, self.fut, self.state = (x.to(device) for x in self.load("train"))
        m = self.state.shape[-1]
        self.perms_t = torch.tensor(self.perms or [list(range(m))], device=device)

    def encode_src(self, encoder, src: torch.Tensor) -> torch.Tensor:
        """Features (..., d) of sources (..., *src), rendered and encoded ENCODE_CHUNK clips at a time."""
        lead = src.shape[: src.dim() - self.src_dims]
        flat = src.reshape(-1, *src.shape[len(lead) :])
        out = torch.cat([encode_clips(encoder, self.render(c), self.cfg.encoder) for c in flat.split(ENCODE_CHUNK)])
        return out.reshape(*lead, -1)

    def readout_dim(self) -> int:
        """Readout outputs per sample; default: the state itself (regression)."""
        return self.state.shape[-1]

    def readout_fit_loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return readout_loss(out, target, self.scale, self.perms_t)

    def decode(self, out: torch.Tensor) -> torch.Tensor:
        """Readout outputs (..., readout_dim) -> states (..., m)."""
        return out

    def batch(self, size: int) -> torch.Tensor:
        return torch.randint(len(self.obs), (size,)).to(self.device)

    def encode(self, encoder, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_src(encoder, self.obs[idx]), self.encode_src(encoder, self.fut[idx])

    def fit_readout(self, cfg, encoder, out_dir, log) -> StateReadout:
        reseed(cfg.seed, READOUT_SEED_OFFSET)
        readout = StateReadout(cfg.encoder.d, self.readout_dim()).to(self.device)
        path = out_dir / "readout.pt"
        if path.exists():
            readout.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            return readout
        with torch.no_grad():  # frozen encoder: every train observation and future encoded once
            h_all, t_all = self.encode_src(encoder, self.obs), self.encode_src(encoder, self.fut)
        r, n_fut = cfg.readout, self.fut.shape[1]
        opt = torch.optim.AdamW(readout.parameters(), lr=r.lr, weight_decay=r.weight_decay)
        for it in range(1, r.iters + 1):
            reseed(cfg.seed, READOUT_SEED_OFFSET + it)
            i = torch.randint(len(self.obs), (r.batch,)).to(self.device)
            j = torch.randint(n_fut, (r.batch,)).to(self.device)
            loss = self.readout_fit_loss(readout(h_all[i], t_all[i, j]), self.state[i, j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), r.clip)
            opt.step()
            if it % 50 == 0:
                log.log({"readout_loss": loss.item(), "readout_it": it})
        save_atomic(readout.state_dict(), path)
        return readout

    def post_readout(self, cfg, groups, readout, out_dir, log) -> None:
        pass

    @torch.no_grad()
    def evaluate(self, cfg, encoder, predictor, prior, readout) -> dict[str, float]:
        obs, fut, state = (x.to(self.device) for x in self.load(cfg.eval.split))
        n, k = min(cfg.eval.n_obs, len(obs)), cfg.eval.k_eval
        obs, fut, state = obs[:n], fut[:n], state[:n]
        h = self.encode_src(encoder, obs)  # (n, d)
        pred = predictor(h, prior.sample(h, k))  # (n, K, d)
        dec = self.decode(readout(h[:, None].expand(-1, k, -1), pred)).cpu().double().numpy()
        truth, perms = state.cpu().double().numpy(), self.perms_t.cpu().numpy()
        per_obs = [futures_metrics(dec[i], truth[i], self.scale, perms, self.max_radius) for i in range(n)]
        metrics = {key: float(np.mean([m[key] for m in per_obs])) for key in per_obs[0]}

        # Readout-only diagnostic: decode each encoded true future with the same validity rule. Its rate is the
        # ceiling the world model's validity can reach through this readout.
        n_fut = fut.shape[1]
        rd = readout(h[:, None].expand(-1, n_fut, -1), self.encode_src(encoder, fut))
        rd = self.decode(rd).cpu().double().numpy()
        own = []
        for i in range(n):
            valid, nearest, _ = match_futures(rd[i], truth[i], self.scale, perms, self.max_radius)
            own.append(valid & (nearest == np.arange(n_fut)))
        metrics["readout_validity"] = float(np.mean(own))
        return metrics
