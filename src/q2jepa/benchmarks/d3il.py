"""D3IL Avoiding (Q-JEPA §5.4, App. B) on rendered segment clips.

Stage 1 trains the world model on observation -> next-segment pairs (one target each). Stage 2 fits the spline
readout on cached frozen features. Stage 3 (post_readout) trains predictor and prior only, on windows of three
consecutive real pairs. Evaluation rolls out from the common stationary start and scores accepted paths' routes
(App. B.4).
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F

from q2jepa.benchmarks.mmnist import encode_clips
from q2jepa.data import d3il
from q2jepa.losses import energy_score
from q2jepa.metrics import ROUTES, expected_route_stats
from q2jepa.readout.d3il import SplineReadout
from q2jepa.seeding import FINETUNE_SEED_OFFSET, READOUT_SEED_OFFSET, reseed, save_atomic

ENCODE_CHUNK = 128  # clips per encoder call: bounds HQViT's CUDA-Q register


def _encode(encoder, pts: torch.Tensor, e) -> torch.Tensor:
    return torch.cat([encode_clips(encoder, d3il.render_segment(c), e) for c in pts.split(ENCODE_CHUNK)])


def to_metres(pts: torch.Tensor) -> torch.Tensor:
    return pts * d3il.NORM_SCALE + torch.tensor(d3il.NORM_CENTRE, dtype=pts.dtype, device=pts.device)


def _segment_hits_circle(a: np.ndarray, b: np.ndarray, centre: np.ndarray, radius: float) -> bool:
    """Whether any edge a[i] -> b[i] passes within `radius` of `centre` (continuous check)."""
    ab, ac = b - a, centre - a
    t = np.clip((ac * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-18), 0, 1)
    return bool((np.linalg.norm(a + t[:, None] * ab - centre, axis=-1) < radius).any())


def accept(path: np.ndarray) -> int | None:
    """Route 0..23 of an accepted path (metres, start first, already cut at the goal), else None (App. B.4).

    Accepted: reaches the goal line, crosses every obstacle row upward, no edge comes within an obstacle radius +
    1 cm of its centre, and every point stays within the reach envelope.
    """
    if path[-1, 1] < d3il.GOAL_Y - 1e-12:
        return None
    route = d3il.route_label(path)
    if route is None:
        return None
    a, b = path[:-1], path[1:]
    for row_y, xs, radius in zip(d3il.ROW_Y, d3il.ROW_OBSTACLE_X, d3il.ROW_RADIUS, strict=True):
        for x in xs:
            if _segment_hits_circle(a, b, np.array([x, row_y]), radius + d3il.INFLATION):
                return None
    if (np.linalg.norm(path - np.array(d3il.REACH_BASE), axis=-1) > d3il.REACH + d3il.REACH_TOL).any():
        return None
    return route


class D3ILBenchmark:
    def __init__(self, cfg, device: torch.device):
        self.cfg, self.device = cfg, device
        ds = d3il.D3ILAvoiding("train", cfg.dataset.fold, cfg.dataset.root)
        items = [ds[i] for i in range(len(ds))]
        self.obs = torch.stack([x["obs_pts"] for x in items]).to(device)  # (n, 24, 2)
        self.fut = torch.stack([x["fut_pts"] for x in items]).to(device)
        demo = np.array([int(x["demo_id"]) for x in items])
        # windows of three consecutive pairs from one demo (items are in demo, segment order)
        self.windows = torch.as_tensor(np.flatnonzero((demo[:-2] == demo[1:-1]) & (demo[1:-1] == demo[2:])))
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def batch(self, size: int) -> torch.Tensor:
        return torch.randint(len(self.obs), (size,)).to(self.device)

    def encode(self, encoder, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.cfg.encoder
        return _encode(encoder, self.obs[idx], e), _encode(encoder, self.fut[idx], e)

    def _features(self, encoder) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen-encoder features of every train pair, computed once (encoder fixed after stage 1)."""
        if self._cache is None:
            with torch.no_grad():
                self._cache = self.encode(encoder, torch.arange(len(self.obs), device=self.device))
        return self._cache

    def fit_readout(self, cfg, encoder, out_dir, log) -> SplineReadout:
        reseed(cfg.seed, READOUT_SEED_OFFSET)
        readout = SplineReadout(cfg.encoder.d).to(self.device)
        path = out_dir / "readout.pt"
        if path.exists():
            readout.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            return readout
        h, t = self._features(encoder)
        r = cfg.readout
        opt = torch.optim.AdamW(readout.parameters(), lr=r.lr, weight_decay=r.weight_decay)
        for it in range(1, r.iters + r.iters_low + 1):
            reseed(cfg.seed, READOUT_SEED_OFFSET + it)
            if it == r.iters + 1:
                for g in opt.param_groups:
                    g["lr"] = r.lr_low
            i = torch.randint(len(self.obs), (r.batch,)).to(self.device)
            loss = F.smooth_l1_loss(readout(h[i], t[i]), self.fut[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), r.clip)
            opt.step()
            if it % 50 == 0:
                log.log({"readout_loss": loss.item(), "readout_it": it})
        save_atomic(readout.state_dict(), path)
        return readout

    def post_readout(self, cfg, groups, readout, out_dir, log) -> None:
        """Stage 3 (App. B.3): predictor and prior only, on real windows; encoder and readout stay frozen."""
        from q2jepa.train import _prior_loss

        t = cfg.trainer
        if not t.finetune_iters:
            return
        predictor, prior = groups["predictor"][0], groups["prior"][0]
        path = out_dir / "finetune.pt"
        if path.exists():
            ck = torch.load(path, map_location=self.device, weights_only=True)
            predictor.load_state_dict(ck["predictor"])
            prior.load_state_dict(ck["prior"])
            return
        h_all, t_all = self._features(groups["encoder"][0])
        predictor.requires_grad_(True)
        prior.requires_grad_(getattr(prior, "trainable", True))
        params = [p for m in (predictor, prior) for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=t.lr_finetune, weight_decay=t.weight_decay)
        dist = (t.energy_beta, t.energy_eps)
        for it in range(1, t.finetune_iters + 1):
            reseed(cfg.seed, FINETUNE_SEED_OFFSET + it)
            starts = self.windows[torch.randint(len(self.windows), (t.finetune_windows,))].to(self.device)
            i = (starts[:, None] + torch.arange(3, device=self.device)).reshape(-1)  # 3 consecutive pairs each
            h, tgt = h_all[i], t_all[i]
            z = prior.sample(h, t.k_train)
            pred = predictor(h, z)
            loss = energy_score(pred, tgt, *dist) + _prior_loss(cfg, prior, predictor, h, z, pred.detach(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for m in (predictor, prior):
                torch.nn.utils.clip_grad_norm_(m.parameters(), t.clip)
            opt.step()
            if it % 50 == 0:
                log.log({"finetune_loss": loss.item(), "finetune_it": it})
        predictor.requires_grad_(False)
        prior.requires_grad_(False)
        save_atomic({"predictor": predictor.state_dict(), "prior": prior.state_dict()}, path)

    @torch.no_grad()
    def rollouts(self, cfg, encoder, predictor, prior, readout) -> list[np.ndarray]:
        """cfg.eval.rollouts paths (metres) from the stationary start, each cut at its first goal crossing."""
        e, paths = cfg.encoder, []
        start = torch.tensor(d3il.normalize(d3il.start_point()), dtype=torch.float32, device=self.device)
        for n in torch.arange(cfg.eval.rollouts).split(128):  # App. B.4: 128-path repetitions
            pts = start.expand(len(n), 24, 2)
            segs, done = [], torch.zeros(len(n), dtype=torch.bool, device=self.device)
            for _ in range(cfg.eval.max_calls):
                h = _encode(encoder, pts, e)
                pts = readout(h, predictor(h, prior.sample(h, 1))[:, 0])
                segs.append(pts)
                done |= (to_metres(pts)[..., 1] >= d3il.GOAL_Y).any(-1)
                if done.all():
                    break
            full = to_metres(torch.cat([start.expand(len(n), 1, 2), *segs], 1)).cpu().double().numpy()
            paths += [d3il.to_goal(p) for p in full]
        return paths

    @torch.no_grad()
    def evaluate(self, cfg, encoder, predictor, prior, readout) -> dict[str, float]:
        if cfg.eval.split != "test":
            # Rollouts from the common start are the reported metric; there is no held-out dev signal (spec §5.6).
            raise ValueError("D3IL has no dev split: evaluate with eval.split=test; reuse MovingMNIST-tuned rates")
        routes = [accept(p) for p in self.rollouts(cfg, encoder, predictor, prior, readout)]
        accepted = np.array([r for r in routes if r is not None], dtype=np.int64)
        counts = np.bincount(accepted, minlength=ROUTES)
        covered, tv = expected_route_stats(counts, cfg.eval.subset)
        if len(accepted) < cfg.eval.subset:
            warnings.warn(
                f"only {len(accepted)} accepted paths (< eval.subset={cfg.eval.subset}): route stats use all of them",
                stacklevel=2,
            )
        # Readout-only diagnostic on the train demos (render check; the fold's eval demos stay untouched).
        # Fresh features: evaluation runs the CUDA-Q circuits, while the cache came from the training simulator.
        i = torch.arange(min(cfg.eval.n_obs, len(self.obs)), device=self.device)
        h, t = self.encode(encoder, i)
        err = (to_metres(readout(h, t)) - to_metres(self.fut[i])).norm(dim=-1).mean()
        return {
            "routes": covered,
            "coverage": covered / ROUTES,
            "tv": tv,
            "acceptance": len(accepted) / len(routes),
            "n_accepted": float(len(accepted)),
            "subset": float(min(cfg.eval.subset, len(accepted))),
            "readout_ade_m": float(err),
        }
