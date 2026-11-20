"""Stochastic MovingMNIST (Q-JEPA §5.2, App. A.2, F.2, F.4): translation readout and direction metrics."""

import numpy as np
import torch

from q2jepa.data.mmnist import StochasticMovingMNIST
from q2jepa.encoder import prepare
from q2jepa.metrics import mmnist_metrics
from q2jepa.readout.mmnist import TranslationReadout, readout_loss, render
from q2jepa.seeding import READOUT_SEED_OFFSET, reseed, save_atomic


def encode_clips(encoder, x: torch.Tensor, e) -> torch.Tensor:
    return encoder(prepare(x, e.res, e.input_norm, e.input_scale))


class MovingMNISTBenchmark:
    def __init__(self, cfg, device: torch.device):
        self.cfg, self.device = cfg, device
        self.train = StochasticMovingMNIST("train", cfg.dataset.root, n_obs=cfg.dataset.n_train)

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        items = [self.train[i] for i in torch.randint(len(self.train), (size,)).tolist()]
        obs = torch.stack([x["obs"] for x in items]).to(self.device)
        fut = torch.stack([x["fut"] for x in items]).to(self.device)
        return obs, fut

    def encode(self, encoder, batch) -> tuple[torch.Tensor, torch.Tensor]:
        obs, fut = batch
        return encode_clips(encoder, obs, self.cfg.encoder), encode_clips(encoder, fut, self.cfg.encoder)

    def fit_readout(self, cfg, encoder, out_dir, log) -> TranslationReadout:
        reseed(cfg.seed, READOUT_SEED_OFFSET)  # readout init must not depend on whether training ran this session
        readout = TranslationReadout(cfg.encoder.d).to(self.device)
        path = out_dir / "readout.pt"
        if path.exists():
            readout.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            return readout
        r = cfg.readout
        opt = torch.optim.AdamW(readout.parameters(), lr=r.lr, weight_decay=r.weight_decay)
        for it in range(1, r.iters + 1):
            reseed(cfg.seed, READOUT_SEED_OFFSET + it)
            obs, fut = self.batch(r.batch)
            with torch.no_grad():
                h, tgt = self.encode(encoder, (obs, fut))
            loss = readout_loss(render(obs[:, 2], readout(h, tgt)), fut)
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
        split = cfg.eval.split
        n_split = cfg.dataset.n_test if split == "test" else cfg.dataset.n_dev
        eval_obs = StochasticMovingMNIST(split, cfg.dataset.root, n_obs=n_split).unique_obs()
        obs = eval_obs[: cfg.eval.n_obs].to(self.device)
        k = cfg.eval.k_eval
        h = encode_clips(encoder, obs, cfg.encoder)
        pred = predictor(h, prior.sample(h, k))  # (B, K, d)
        disp = readout(h.repeat_interleave(k, 0), pred.reshape(-1, pred.shape[-1])).reshape(len(obs), k, 3, 2)
        per_obs = [mmnist_metrics(d) for d in disp.cpu().double().numpy()]
        return {key: float(np.mean([m[key] for m in per_obs])) for key in per_obs[0]}
