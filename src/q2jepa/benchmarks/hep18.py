"""HEP18 (Q-JEPA §5.3, App. A.2, F.3, F.4) on rendered per-car clips.

Each car's 3-frame clip goes through the shared encoder (encoder.d = 16); the four outputs concatenated in N, W, S, E
order are the scene h (64). Targets are all 18 joint futures. A car has only 4 distinct futures (schedules A, B, C,
D), so each observation encodes 16 future clips and the 18 scene targets are assembled from them (exact).
"""

import numpy as np
import torch

from q2jepa.benchmarks.mmnist import encode_clips
from q2jepa.data.hep18 import _SCHEDULES, HEP18, render_agent_clip
from q2jepa.metrics import _joint_valid, hep18_metrics
from q2jepa.readout.hep18 import SpeedReadout, integrate, readout_loss
from q2jepa.seeding import READOUT_SEED_OFFSET, reseed, save_atomic

AGENTS, SCHEDULES = 4, 4
ENCODE_CHUNK = 128  # clips per encoder call: bounds HQViT's CUDA-Q register (log2(128) batch qubits)

# FIRST_FUTURE[c, k]: a future j in which car c follows schedule k (each pair occurs).
FIRST_FUTURE = np.array([[int(np.flatnonzero(_SCHEDULES[:, c] == k)[0]) for k in range(SCHEDULES)] for c in range(4)])


class _Split:
    """Tensors of one split on the device: obs (n, 4, 3, 2), per-car distinct futures (n, 4, 4, 3, 2), all futures."""

    def __init__(self, split: str, root: str, device: torch.device):
        ds = HEP18(split, root)
        self.obs = torch.tensor(ds.obs_state, dtype=torch.float32, device=device)
        self.fut = torch.tensor(ds.fut_state, dtype=torch.float32, device=device)  # (n, 18, 4, 3, 2)
        cars = np.arange(AGENTS)[:, None]
        self.uniq = self.fut[:, torch.as_tensor(FIRST_FUTURE), torch.as_tensor(cars)]  # (n, 4, 4, 3, 2)

    def __len__(self) -> int:
        return len(self.obs)


def _encode_cars(encoder, s: torch.Tensor, e) -> torch.Tensor:
    """Features (..., d_agent) of per-car clips rendered from positions s (..., 3), in chunks."""
    flat = s.reshape(-1, 3)
    out = [encode_clips(encoder, render_agent_clip(c), e) for c in flat.split(ENCODE_CHUNK)]
    return torch.cat(out).reshape(*s.shape[:-1], -1)


def scene_targets(uniq_feat: torch.Tensor) -> torch.Tensor:
    """(B, 4 cars, 4 schedules, d_agent) -> the 18 scene targets (B, 18, 4 * d_agent)."""
    sched = torch.as_tensor(_SCHEDULES, device=uniq_feat.device)  # (18, 4)
    return uniq_feat[:, torch.arange(AGENTS, device=uniq_feat.device)[None], sched].flatten(-2)


class HEP18Benchmark:
    def __init__(self, cfg, device: torch.device):
        if cfg.dataset.agents != AGENTS:
            raise ValueError("HEP18 needs dataset.agents=4 (use +experiment=hep18)")
        self.cfg, self.device = cfg, device
        self.train = _Split("train", cfg.dataset.root, device)

    def batch(self, size: int) -> torch.Tensor:
        return torch.randint(len(self.train), (size,)).to(self.device)

    def encode(self, encoder, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.cfg.encoder
        h = _encode_cars(encoder, self.train.obs[idx, ..., 0], e).flatten(-2)  # (B, 64)
        return h, scene_targets(_encode_cars(encoder, self.train.uniq[idx, ..., 0], e))

    def _readout_step(self, readout, feat, obs, fut):
        """feat (N, 4, d_agent), obs (N, 4, 3, 2), fut (N, 4, 3, 2) -> loss."""
        speeds = readout(feat)
        s = integrate(obs[..., 2, 0], obs[..., 2, 1], speeds)
        return readout_loss(s, speeds, fut[..., 0], fut[..., 1])

    def fit_readout(self, cfg, encoder, out_dir, log) -> SpeedReadout:
        reseed(cfg.seed, READOUT_SEED_OFFSET)
        readout = SpeedReadout(cfg.encoder.d).to(self.device)
        path = out_dir / "readout.pt"
        if path.exists():
            readout.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            return readout
        with torch.no_grad():  # the frozen encoder's features of every distinct train future, computed once
            cache = _encode_cars(encoder, self.train.uniq[..., 0], cfg.encoder)  # (n, 4, 4, d_agent)
        r = cfg.readout
        opt = torch.optim.AdamW(readout.parameters(), lr=r.lr, weight_decay=r.weight_decay)
        sched = torch.as_tensor(_SCHEDULES, device=self.device)
        cars = torch.arange(AGENTS, device=self.device)[None]
        for it in range(1, r.iters + 1):
            reseed(cfg.seed, READOUT_SEED_OFFSET + it)
            i = torch.randint(len(self.train), (r.batch,)).to(self.device)
            j = torch.randint(18, (r.batch,)).to(self.device)
            feat = cache[i[:, None], cars, sched[j]]  # (N, 4, d_agent)
            loss = self._readout_step(readout, feat, self.train.obs[i], self.train.fut[i, j])
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

    def decode(self, readout, feat: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scene features (..., 64) and obs (..., 4, 3, 2) broadcastable -> positions, speeds (..., 4, 3)."""
        speeds = readout(feat.unflatten(-1, (AGENTS, -1)))
        return integrate(obs[..., 2, 0], obs[..., 2, 1], speeds), speeds

    @torch.no_grad()
    def evaluate(self, cfg, encoder, predictor, prior, readout) -> dict[str, float]:
        data = _Split(cfg.eval.split, cfg.dataset.root, self.device)
        n, k = min(cfg.eval.n_obs, len(data)), cfg.eval.k_eval
        obs, fut = data.obs[:n], data.fut[:n]
        h = _encode_cars(encoder, obs[..., 0], cfg.encoder).flatten(-2)  # (n, 64)
        pred = predictor(h, prior.sample(h, k))  # (n, K, 64)
        s, v = self.decode(readout, pred, obs[:, None])
        s, v, s_true, v_true = (x.cpu().double().numpy() for x in (s, v, fut[..., 0], fut[..., 1]))
        rng = np.random.default_rng(0)
        per_obs = [hep18_metrics(s[i], v[i], s_true[i], v_true[i], rng) for i in range(n)]
        metrics = {key: float(np.mean([m[key] for m in per_obs])) for key in per_obs[0]}

        # Readout-only diagnostic (App. F.4): decode the encoded real futures; each must match its own future.
        tgt = scene_targets(_encode_cars(encoder, data.uniq[:n, ..., 0], cfg.encoder))  # (n, 18, 64)
        rs, rv = (x.cpu().double().numpy() for x in self.decode(readout, tgt, obs[:, None]))
        hits = [_joint_valid(rs[i], rv[i], s_true[i], v_true[i]) for i in range(n)]
        metrics["readout_validity"] = float(np.mean([(ok & (j == np.arange(18))).mean() for ok, j in hits]))
        return metrics
