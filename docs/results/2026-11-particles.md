# Charged particles (negative result, stopped early)

Charged particles `nri` ([candidate datasets spec §4](../superpowers/specs/2026-11-21-q2jepa-candidate-datasets.md)), test
split (128 observations), 128 prior draws each, **seed 0 only**, MovingMNIST's schedules (`+tune=s1`), preset
`+experiment=nri`. Closed on 2026-11-23: no row does useful prediction, so no more seeds were run.

**Reproduce:** `python scripts/run_grid.py --bench nri --stage grid --seeds 0` at the commit in the row (with the
MovingMNIST tuning runs under `--root`; `nri` data is generated). One run is `python -m q2jepa.train
+experiment=nri +tune=s1 eval.split=test encoder=<cnn|vit|hqvit> prior=<softmax_categorical|entangled_qcbm> seed=0
quantum=gpu trainer.device=cuda`, plus `quantum.hqvit_train=torch` for HQViT. Pooling is the preset's at each SHA.
**Platform:** Colab T4, `nvidia` target. One seed: no spread; do not read differences between rows as effects.

| Encoder | Pooling | Prior | SHA | Coverage ↑ | Validity ↑ | TV ↓ | minADE ↓ (px) | ES ↓ (px, descriptive) | Readout validity ↑ |
|---|---|---|---|---|---|---|---|---|---|
| CNN | — | Softmax-Categorical | 9c56c5a | 0.8 | 0.4 | 0.996 | 5.71 | 5.25 | 39.3 |
| CNN | — | Entangled-QCBM | 9c56c5a | 1.8 | 1.1 | 0.989 | 5.69 | 5.25 | 42.0 |
| ViT twin | mean | Softmax-Categorical | 9c56c5a | 0.0 | 0.0 | 1.000 | 6.82 | 6.29 | 2.3 |
| ViT twin | mean | Entangled-QCBM | 9c56c5a | 0.6 | 0.4 | 0.996 | 6.76 | 6.26 | 2.3 |
| HQViT (torch-trained) | mean | Softmax-Categorical | 9c56c5a | 0.0 | 0.0 | 1.000 | 6.74 | 6.63 | 2.9 |
| HQViT (torch-trained) | mean | Entangled-QCBM | 9c56c5a | 0.4 | 0.2 | 0.998 | 6.79 | 6.69 | 2.9 |
| ViT twin | flatten | Softmax-Categorical | eef990e | 0.0 | 0.0 | 1.000 | 6.38 | 5.94 | 9.0 |
| ViT twin | flatten | Entangled-QCBM | eef990e | 0.8 | 0.4 | 0.996 | 6.26 | 5.89 | 9.4 |

HQViT with flatten pooling was stopped before it finished (about iterations 2750 and 3150; checkpoints kept, no
metrics). HQViT here was evaluated through the CUDA-Q circuits.

**Reading.** Every row is at the floor. For the CNN the readout can decode the true future 39–42% of the time, so its
limit is the predicted futures. For ViT and HQViT the readout itself is at 2–9%, so their validity says little about
the prior. Flattened pooling raised the ViT readout from 2% to 9% without raising validity.
