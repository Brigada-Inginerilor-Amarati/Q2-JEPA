# MovingMNIST: CNN encoder × latent priors (phase 2)

Stochastic single-digit MovingMNIST (Q-JEPA App. F.2), test split, 80 observations × 128 prior draws, 3 seeds.
Phase-2 preset: native 64×64, `input_norm=l2`, `input_scale=0.144`; CNN's tuned schedule `+tune=s1` (dev-selected,
phase-2 spec F5), evaluated once on test. QCBMs train with noisy parameter shift (b/B 8/64, r/G 9/9, 256 shots,
0.5% readout flips in training and evaluation); Frozen-QCBM is noise-free and never updated.

**Reproduce:** `OMP_NUM_THREADS=2 uv run python -m q2jepa.train +experiment=phase2 +tune=s1 eval.split=test
encoder=cnn prior=<prior> seed=<0|1|2>` at commit **e145b19**.
**Platform:** Linux x86-64 CPU (20 cores), 12 runs concurrently, 2 threads each, `qpp-cpu` (F9: compare only with
rows from the same platform).

| Prior | Prior params | Coverage ↑ | Validity ↑ | TV ↓ | minADE ↓ (px) | ES ↓ (px) | Coverage per seed |
|---|---|---|---|---|---|---|---|
| Softmax-Categorical | 3,640 | 92.2 ± 4.8 | 96.6 ± 2.5 | 0.155 ± 0.035 | 0.93 ± 0.17 | 9.61 ± 0.03 | 94.2 / 86.7 / 95.6 |
| Entangled-QCBM | 3,423 | 90.4 ± 5.5 | 95.1 ± 1.7 | 0.195 ± 0.024 | 1.10 ± 0.15 | 9.69 ± 0.03 | 93.8 / 84.1 / 93.4 |
| Product-QCBM | 3,423 | 92.6 ± 2.8 | 95.6 ± 2.8 | 0.186 ± 0.016 | 0.96 ± 0.03 | 9.65 ± 0.01 | 90.2 / 91.9 / 95.6 |
| Frozen-QCBM | 0 trainable | 87.8 ± 3.8 | 91.7 ± 1.6 | 0.229 ± 0.038 | 1.21 ± 0.26 | 9.71 ± 0.09 | 91.7 / 87.3 / 84.2 |
| *Paper: Softmax-Categorical* | 3,640 | 93.6 ± 0.5 | 95.8 ± 0.5 | 0.145 ± 0.002 | 0.79 ± 0.04 | 9.58 ± 0.02 | |
| *Paper: Entangled-QCBM* | 3,423 | 94.0 ± 2.3 | 96.5 ± 1.1 | 0.160 ± 0.024 | 0.84 ± 0.11 | 9.61 ± 0.03 | |
| *Paper: Product-QCBM* | 3,423 | 94.3 ± 2.4 | 96.2 ± 1.6 | 0.167 ± 0.019 | 0.80 ± 0.08 | 9.60 ± 0.02 | |
| *Paper: Frozen-QCBM* | 0 | 93.3 ± 3.7 | 95.6 ± 1.8 | 0.176 ± 0.027 | 0.84 ± 0.13 | 9.61 ± 0.04 | |

Mean ± sample s.d. over seeds 0–2. Encoder 439,600 params (all rows).

**Reading.** Trained QCBMs are within one seed s.d. of Softmax-Categorical on coverage and validity, with fewer
prior parameters; Softmax-Categorical keeps the best frequency match (TV), as in the paper. Training the Born prior
matters: Frozen-QCBM is lowest on every metric. Product ≈ Entangled is expected here — the target frequencies
factor over bits (paper §5.2). Seed 1 is the weakest seed for every trained prior, so pairing by seed matters.
Our spreads are larger than the paper's, and our TV/minADE are somewhat worse across all priors (normalized input,
different platform); compare within this table, not across to the paper.
