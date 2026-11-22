"""Shared benchmark for whole-scene clips with J equally likely futures (charged particles, chess).

A subclass provides `load(split)` -> obs clips (n, 3, 64, 64), future clips (n, J, 3, 64, 64) and future states
(n, J, m), plus `tol` (m,), the per-dimension match tolerance. The encoder sees whole clips (dataset.agents = 1);
targets are the J encoded futures. The readout decodes [h; t] to the state; metrics match decoded samples to the
J true futures (metrics.futures_metrics). Clips are rendered once and kept on the device.
"""

import numpy as np
import torch

from q2jepa.benchmarks.mmnist import encode_clips
from q2jepa.metrics import futures_metrics
from q2jepa.readout.futures import StateReadout, readout_loss
from q2jepa.seeding import READOUT_SEED_OFFSET, reseed, save_atomic

ENCODE_CHUNK = 128  # clips per encoder call: bounds HQViT's CUDA-Q register, as in HEP18


def encode_chunked(encoder, clips: torch.Tensor, e) -> torch.Tensor:
    """Features (..., d) of clips (..., 3, 64, 64)."""
    flat = clips.reshape(-1, *clips.shape[-3:])
    out = torch.cat([encode_clips(encoder, c, e) for c in flat.split(ENCODE_CHUNK)])
    return out.reshape(*clips.shape[:-3], -1)


class FuturesBenchmark:
    tol: tuple[float, ...] = ()

    def load(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __init__(self, cfg, device: torch.device):
        if cfg.dataset.agents != 1:
            raise ValueError(f"{cfg.dataset.name} encodes whole scenes: dataset.agents must be 1")
        self.cfg, self.device = cfg, device
        self.obs, self.fut, self.state = (x.to(device) for x in self.load("train"))
        self.tol_t = torch.tensor(self.tol, dtype=torch.float32, device=device)

    def batch(self, size: int) -> torch.Tensor:
        return torch.randint(len(self.obs), (size,)).to(self.device)

    def encode(self, encoder, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.cfg.encoder
        return encode_chunked(encoder, self.obs[idx], e), encode_chunked(encoder, self.fut[idx], e)

    def fit_readout(self, cfg, encoder, out_dir, log) -> StateReadout:
        reseed(cfg.seed, READOUT_SEED_OFFSET)
        readout = StateReadout(cfg.encoder.d, self.state.shape[-1]).to(self.device)
        path = out_dir / "readout.pt"
        if path.exists():
            readout.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            return readout
        with torch.no_grad():  # frozen encoder: every train observation and future encoded once
            h_all = encode_chunked(encoder, self.obs, cfg.encoder)
            t_all = encode_chunked(encoder, self.fut, cfg.encoder)
        r, n_fut = cfg.readout, self.fut.shape[1]
        opt = torch.optim.AdamW(readout.parameters(), lr=r.lr, weight_decay=r.weight_decay)
        for it in range(1, r.iters + 1):
            reseed(cfg.seed, READOUT_SEED_OFFSET + it)
            i = torch.randint(len(self.obs), (r.batch,)).to(self.device)
            j = torch.randint(n_fut, (r.batch,)).to(self.device)
            loss = readout_loss(readout(h_all[i], t_all[i, j]), self.state[i, j], self.tol_t)
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
        h = encode_chunked(encoder, obs, cfg.encoder)  # (n, d)
        pred = predictor(h, prior.sample(h, k))  # (n, K, d)
        dec = readout(h[:, None].expand(-1, k, -1), pred).cpu().double().numpy()
        truth, tol = state.cpu().double().numpy(), np.asarray(self.tol)
        per_obs = [futures_metrics(dec[i], truth[i], tol) for i in range(n)]
        metrics = {key: float(np.mean([m[key] for m in per_obs])) for key in per_obs[0]}

        # Readout-only diagnostic: decode each encoded true future; it must match its own future.
        n_fut = fut.shape[1]
        rd = readout(h[:, None].expand(-1, n_fut, -1), encode_chunked(encoder, fut, cfg.encoder))
        own = ((rd - state).abs() / self.tol_t).amax(-1) <= 1.0
        metrics["readout_validity"] = float(own.float().mean())
        return metrics
