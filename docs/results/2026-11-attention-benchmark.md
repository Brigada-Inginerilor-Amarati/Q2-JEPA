# Swap-test attention: CPU vs GPU (2026-11-24)

One attention map per image through the CUDA-Q kernels, ms per image. Commit 14cdde7, Colab VM with a T4: `qpp-cpu` on its
2 vCPUs, `nvidia` on the T4, same session.

`python scripts/bench_attention.py --target <qpp-cpu|nvidia> --res <8|16|32|64> --images 8`
(2 images for `qpp-cpu` at 32; 64 was not run on the CPU). One warm-up call per row, one run each, no error bars.

| res | mode | qubits | qpp-cpu | nvidia (T4) | CPU / GPU |
|---|---|---|---|---|---|
| 8 | factored | 4 | 12.5 | 30.7 | 0.4 |
| 16 | factored | 4 | 325 | 113 | 2.9 |
| 32 | factored | 4 | 1,143 | 480 | 2.4 |
| 64 | factored | 4 | not run | 2,000 | |
| 8 | full | 13 | 217 | 5.3 | 41 |
| 16 | full | 17 | 4,392 | 6.9 | 636 |
| 32 | full | 21 | 83,768 | 45.3 | 1,848 |
| 64 | full | 25 | not run | 559 | |

The full swap test gains more from the GPU as the register grows. The factored loop is many tiny circuits: 2–3× faster at
16–32, 2.5× slower at 8. **The ratios mix precisions:** default targets are fp32 on `nvidia` and fp64 on `qpp-cpu`, so part
of the gain is precision; two CPU cores are also a weak baseline. CUDA-Q 0.16.0. Phase-2 spec §6 timings (2026-11-03,
16 images): 2,100× and 38.5 ms at 21 qubits; its factored CPU times were lower (164 and 376 ms), so factored ratios are
not stable across sessions.
