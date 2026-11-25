# Results overview (2026-11-24)

Test split, mean ± sample s.d. over seeds 0–2, 128 prior draws, Colab T4. HQViT trained with the exact torch simulation and
evaluated through the CUDA-Q circuits. Compare rows only within a benchmark. Details, commits and commands are on each page.

| Benchmark | Metric | CNN Softmax / QCBM | ViT twin Softmax / QCBM | HQViT Softmax / QCBM | Reference |
|---|---|---|---|---|---|
| [MovingMNIST](2026-11-phase2-mmnist-encoders.md) | coverage ↑ | 92.1 ± 6.8 / 91.7 ± 1.6 | 50.1 ± 5.6 / 49.4 ± 3.0 | 35.5 ± 14.3 / 29.0 ± 8.6 | |
| | validity ↑ | 97.0 ± 3.4 / 94.4 ± 1.0 | 57.5 ± 2.1 / 58.8 ± 5.1 | 43.6 ± 16.6 / 36.9 ± 13.8 | |
| [TaxiBJ](2026-11-taxibj.md) | mse_mean ↓ (10⁻³) | 0.946 ± 0.160 / 0.946 ± 0.161 | 1.115 ± 0.144 / 1.133 ± 0.180 | 0.979 ± 0.095 / 0.939 ± 0.085 | copy last map 1.683 |
| [Chess](2026-11-chess.md) | coverage ↑ | 17.8 ± 7.4 / 17.9 ± 5.9 | 9.5 ± 3.3 / 11.5 ± 2.0 | 13.2 ± 2.6 / 11.4 ± 1.4 | random legal move 98.7 |
| | validity ↑ | 9.0 ± 3.7 / 8.0 ± 2.3 | 3.9 ± 1.4 / 4.4 ± 0.6 | 5.7 ± 1.1 / 4.9 ± 0.2 | random legal move 13.1 |
| [Particles](2026-11-particles.md) | validity ↑, seed 0 only, no s.d. | 0.4 / 1.1 | mean pool 0.0 / 0.4; flatten 0.0 / 0.4 | mean pool 0.0 / 0.2; flatten not finished | |

Figures: [MovingMNIST](figures/grid_mmnist.png), [TaxiBJ](figures/grid_taxibj.png), [chess](figures/grid_chess.png)
(`scripts/plot_results.py`). CPU vs GPU timing: [attention benchmark](2026-11-attention-benchmark.md).

**What the results support.**
- Entangled-QCBM is within the seed spread of Softmax-Categorical on every benchmark with three seeds. No quantum advantage.
- HQViT is below its classical twin on MovingMNIST (14–20 coverage points, large seed spread) and not
  worse on TaxiBJ and chess: mostly equal or slightly ahead on means, with 3 seeds and no consistent lead on TaxiBJ (the
  twin wins seed 2). This is no evidence of an advantage; every chess row is below a random legal move.
- TaxiBJ forecasts beat copying the last map by 33–44%.
- Chess and particles are negative results.
