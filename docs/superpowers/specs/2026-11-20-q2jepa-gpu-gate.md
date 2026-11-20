# GPU gate: correctness, timing, precision (2026-11-20)

Results of the HQViT timing gate from [phase-2 spec](2026-11-03-q2jepa-phase2-design.md) §6.

**Platform:** Colab Pro, Tesla T4, ~1.15 units/h. `main` at `0ab636a`, CUDA-Q 0.16, target `nvidia`.
**Not yet measured:** the adjoint HQViT backward (branch `hqvit-adjoint`). These timings use the older
one-circuit-per-angle backward.

## 1. Correctness on GPU

All 37 quantum tests pass on `nvidia` in fp32 and in fp64 (commit `25f21c2`).

Before that commit, 5 tests failed in fp32 because their tolerances assumed fp64 (`qpp-cpu`). They now loosen
tolerances only when `is_fp32()`. Strict finite-difference gradient checks run in fp64 only.

`quantum.precision` (`default` | `fp32` | `fp64`) picks the `nvidia` precision. Non-default values enter the
checkpoint identity. Tests read `Q2JEPA_CUDAQ_PRECISION`.

## 2. Attention benchmark

`scripts/bench_attention.py`, 4×4 patches, ms per image, `nvidia` in fp32.

| Resolution | Variant | Qubits | `qpp-cpu` | `nvidia` (T4) |
|---|---|---|---|---|
| 16×16 | factored | 4 | 122 | 108 |
| 16×16 | full | 17 | 4357 | 6.0 |
| 32×32 | factored | 4 | stopped | 526 |
| 32×32 | full | 21 | stopped | 35.4 |

Each CUDA-Q call costs ~7–8 ms on the GPU regardless of size, so many small circuits are slow. On GPU the
number of calls matters more than the qubit count.

## 3. Timing gate

`scripts/bench_encoder_step.py`, `+experiment=phase2`, batch 64, 1 timed step. Seconds per step; hours per run
counts prediction-stage steps as full steps (upper bound).

| Encoder | Prior | Precision | repr s/it | full s/it | readout s/it | h/run | units/run |
|---|---|---|---|---|---|---|---|
| CNN | Softmax-Categorical | fp32 | 0.013 | 0.023 | 0.002 | 0.03 | 0.0 |
| CNN | Entangled-QCBM | fp32 | 0.011 | 0.187 | 0.002 | 0.20 | 0.2 |
| ViT twin | Softmax-Categorical | fp32 | 0.029 | 0.034 | 0.011 | 0.05 | 0.1 |
| ViT twin | Entangled-QCBM | fp32 | 0.027 | 0.217 | 0.011 | 0.24 | 0.3 |
| HQViT | Softmax-Categorical | fp32 | 24.98 | 21.65 | 0.63 | 24.9 | 28.7 |
| HQViT | Entangled-QCBM | fp32 | 27.90 | 22.40 | 0.76 | 26.0 | 29.9 |
| HQViT | Softmax-Categorical | fp64 | 46.73 | 49.89 | 1.12 | 56.5 | 65.0 |
| HQViT | Entangled-QCBM | fp64 | 53.75 | 54.35 | 1.11 | 61.6 | 70.8 |

- Classical rows cost under 15 minutes per run.
- An HQViT step makes ~800 CUDA-Q calls; call overhead alone is ~6–7 s of the ~22 s.
- fp64 is ~2.3× slower than fp32. The QCBM adds little next to HQViT.

## 4. Precision

Proposal: train in fp32. Circuits are verified in fp64; fp32 adds ~1e-6 relative error, far below shot noise,
readout flips and SGD noise. Precision is part of the platform: every row of a table uses the same value, set
explicitly (`quantum.precision=fp32`).

## 5. Reproduce

On a T4 runtime, run notebook cells 1, 2, 4, 6, then:

```bash
python scripts/bench_encoder_step.py --target nvidia --encoder cnn --encoder vit --encoder hqvit \
    --prior softmax_categorical --prior entangled_qcbm --steps 1 --cu-per-hour 1.15
python scripts/bench_encoder_step.py --target nvidia --encoder hqvit \
    --prior softmax_categorical --prior entangled_qcbm --steps 1 --cu-per-hour 1.15 --set quantum.precision=fp64
```
