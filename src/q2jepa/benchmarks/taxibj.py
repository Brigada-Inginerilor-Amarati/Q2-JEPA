"""TaxiBJ (data/taxibj.py): one future per observation. State: the 3 future inflow/outflow maps in [0, 1]
(3 x 2 x 32 x 32 = 6144 numbers), decoded by the shared [h; t] MLP readout with a mean squared error loss.
Metrics: metrics.flow_metrics (best-of-K and mean MSE/MAE, energy score) plus the readout-only MSE on the encoded
true future, the floor the world model's error can reach through this readout."""

import numpy as np
import torch

from q2jepa.benchmarks.futures import FuturesBenchmark
from q2jepa.data.taxibj import TaxiBJ, render
from q2jepa.metrics import flow_metrics


class TaxiBJBenchmark(FuturesBenchmark):
    src_dims, scale = 4, 1.0  # source: maps (frames, 2, 32, 32)

    def load(self, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ds = TaxiBJ(split, self.cfg.dataset.root)
        fut = ds.fut[:, None]  # (n, 1, 3, 2, 32, 32): J = 1
        return ds.obs, fut, fut.flatten(2)

    def render(self, src: torch.Tensor) -> torch.Tensor:
        return render(src)  # (..., 3, 2, 32, 32) -> (..., 3, 64, 64)

    def readout_fit_loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((out - target.float()) ** 2).mean()

    @torch.no_grad()
    def evaluate(self, cfg, encoder, predictor, prior, readout) -> dict[str, float]:
        obs, fut, state = (x.to(self.device) for x in self.load(cfg.eval.split))
        n, k = min(cfg.eval.n_obs, len(obs)), cfg.eval.k_eval
        obs, fut, state = obs[:n], fut[:n], state[:n, 0].float()
        per_obs, ro = [], []
        for i in range(0, n, 16):  # 16 observations x K samples x 6144 values per call
            sl = slice(i, min(i + 16, n))
            h = self.encode_src(encoder, obs[sl])
            pred = predictor(h, prior.sample(h, k))
            dec = readout(h[:, None].expand(-1, k, -1), pred).cpu().double().numpy()
            truth = state[sl].cpu().double().numpy()
            per_obs += [flow_metrics(d, t) for d, t in zip(dec, truth, strict=True)]
            rd = readout(h, self.encode_src(encoder, fut[sl, 0]))
            ro.append(((rd - state[sl]) ** 2).mean(-1).cpu())
        metrics = {key: float(np.mean([m[key] for m in per_obs])) for key in per_obs[0]}
        metrics["readout_mse"] = float(torch.cat(ro).mean())
        return metrics
