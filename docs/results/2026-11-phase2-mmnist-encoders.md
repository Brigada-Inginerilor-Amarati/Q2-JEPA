# MovingMNIST: CNN vs ViT twin vs HQViT (phase 2, stage B)

Stochastic single-digit MovingMNIST, test split. All three encoders use the same preset: native 64×64,
`input_norm=l2`, 8×8 patches for the twin and HQViT, mean pooling (spec §5.6). Each encoder uses its own
dev-tuned schedule, which was `+tune=s1` for all three (F5, stage A). HQViT trains with
`quantum.hqvit_train=torch`, the exact torch simulation, and is evaluated through the CUDA-Q circuits.
The Entangled-QCBM uses the default noisy parameter shift.

**Reproduce:** `python scripts/run_grid.py --bench mmnist --stage grid --jobs 3` at commit **b7ccae9**. One run
is `python -m q2jepa.train +experiment=phase2 +tune=s1 eval.split=test encoder=<cnn|vit|hqvit> prior=<prior>
seed=<0|1|2> quantum=gpu trainer.device=cuda`, plus `quantum.hqvit_train=torch` for HQViT.
**Platform:** one Colab T4, 3 runs at once, `nvidia` target, Daria's account (F9: compare only with rows from
the same platform).

| Encoder | Prior | Coverage ↑ | Validity ↑ | TV ↓ | minADE ↓ (px) | ES ↓ (px) | Coverage per seed |
|---|---|---|---|---|---|---|---|
| CNN | Softmax-Categorical | 92.1 ± 6.8 | 97.0 ± 3.4 | 0.150 ± 0.047 | 0.92 ± 0.38 | 9.59 ± 0.08 | 100.0 / 88.3 / 88.1 |
| CNN | Entangled-QCBM | 91.7 ± 1.6 | 94.4 ± 1.0 | 0.196 ± 0.016 | 1.07 ± 0.19 | 9.67 ± 0.06 | 90.5 / 93.4 / 91.1 |
| ViT twin | Softmax-Categorical | 50.1 ± 5.6 | 57.5 ± 2.1 | 0.542 ± 0.023 | 2.85 ± 0.17 | 10.25 ± 0.03 | 48.1 / 45.8 / 56.4 |
| ViT twin | Entangled-QCBM | 49.4 ± 3.0 | 58.8 ± 5.1 | 0.524 ± 0.018 | 2.76 ± 0.12 | 10.19 ± 0.07 | 52.5 / 49.2 / 46.6 |
| HQViT (torch-trained) | Softmax-Categorical | 35.5 ± 14.3 | 43.6 ± 16.6 | 0.671 ± 0.119 | 3.58 ± 0.59 | 11.00 ± 0.50 | 50.5 / 34.1 / 22.0 |
| HQViT (torch-trained) | Entangled-QCBM | 29.0 ± 8.6 | 36.9 ± 13.8 | 0.733 ± 0.092 | 3.92 ± 0.66 | 11.18 ± 0.60 | 38.9 / 23.1 / 25.0 |

Mean ± sample s.d. over seeds 0–2.

**Reading.**
- CNN passes the classical acceptance bar (coverage ≥ 80, validity ≥ 85, TV ≤ 0.25) with both priors.
- Entangled-QCBM matches Softmax-Categorical within one seed s.d. for every encoder.
- HQViT is below its classical twin (F4) by 14–20 coverage points, and its seed spread is 3–5× the twin's.
- Both ViT-style encoders are far below the CNN anchor. The twin gap was present on dev before any quantum part was
  involved, and neither higher learning rates nor soft-argmax pooling closed it (spec §5.6).
- F5 note: the twin and HQViT saw 2 poolings × 3 schedules on dev; the CNN saw 3 schedules.
