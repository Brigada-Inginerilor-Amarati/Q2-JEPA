# Classical MovingMNIST baselines (kickstart)

Stochastic single-digit MovingMNIST (Q-JEPA App. F.2), CNN encoder, Q-JEPA Table 5 settings, seed 0,
80 test observations × 128 prior draws. Trained on a MacBook CPU (~20 min per run).
Reproduce with `uv run python -m q2jepa.train <overrides> seed=0`.

| Run | Overrides | SHA | Coverage ↑ | Validity ↑ | TV ↓ | minADE ↓ (px) | ES ↓ (px) |
|---|---|---|---|---|---|---|---|
| Softmax-Categorical, 64×64 | `encoder.res=64` | 747bbda | 94.8 | 97.0 | 0.140 | 0.78 | 9.58 |
| Softmax-Categorical, 64×64 (acceptance) | `encoder.res=64` | cd0d8af | 94.8 | 96.8 | 0.142 | 0.82 | 9.59 |
| Softmax-Categorical, 16×16 | `encoder.res=16` | 747bbda | 90.8 | 95.8 | 0.189 | 1.11 | 9.73 |
| Mixture-2, 64×64 | `encoder.res=64 prior=mixture` | 747bbda | 91.7 | 92.8 | 0.174 | 0.95 | 9.62 |
| *Paper: Softmax-Categorical* | (3-seed mean) | — | 93.6 | 95.8 | 0.145 | 0.79 | 9.58 |
| *Paper: Mixture-2* | (3-seed mean) | — | 88.5 | 95.8 | 0.186 | 0.98 | 9.63 |

Acceptance criteria (plan Task 10): validity ≥ 85%, coverage ≥ 80%, TV ≤ 0.25. **Met.**
The cd0d8af run predates the readout-init reseed fix (747bbda), so its readout initialization differs.

16×16 is the HQViT input size (8 qubits per register). Dropping the classical encoder to that resolution
costs ~4 points of coverage and +0.33 px minADE: this is the baseline the quantum encoder must match.
