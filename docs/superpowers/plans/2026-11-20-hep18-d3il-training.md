# HEP18 and D3IL training (2026-11-20)

Wire HEP18 and D3IL into `train.py` as rendered images for every encoder. The readouts, metrics and training
settings follow the Q-JEPA paper (§5.3–5.5, App. A.2, B, C, F.3, F.4). The data and renderers already exist
(`data/hep18.py`, `data/d3il.py`).

## Structure

- `src/q2jepa/benchmarks/{mmnist,hep18,d3il}.py`: one class per benchmark. Each class handles training batches,
  scene encoding, targets, predictor choice, readout fitting and evaluation. `train.py` keeps the single loop and
  calls into these classes. MovingMNIST moves behind the same interface, and its metrics must stay byte-identical
  (regression test).
- Readouts live in `readout/hep18.py` and `readout/d3il.py`; metrics live in `metrics.py`.
- Presets `conf/experiment/hep18.yaml` and `conf/experiment/d3il.yaml` (Table 5, App. B) select the dataset and
  set its own `input_scale`. Every encoder uses the same preset.

## Shared changes

| Change | Where |
|---|---|
| Multiple targets per observation, `tgt (B, J, d)`; MovingMNIST and D3IL use J = 1 | `losses.py`, `train.py` |
| Smoothed distance ρ_β,ε (Eq. 18); β = 1, ε = 0 is the current distance | `losses.py`, `trainer.energy_beta`, `trainer.energy_eps` |
| Change regularizer λΔ R(t − h), averaged over the J views | `trainer.lambda_change` |
| QCBM CNOT layout `brickwork` or `hep18_grid` (App. C) | `prior.layout`, `quantum/born.py` |
| Predictor kinds: `shared` (now), `agent_local` (HEP18), `scaled` (D3IL, Eq. 15) | `predictor.py`, `predictor.kind` |

## HEP18

- **Input:** one clip per car (`render_agent_clip`); a shared encoder with `d = 16`; 4 cars concatenated into
  h ∈ ℝ⁶⁴. Each car has only 4 distinct future clips, so each observation encodes 16 future clips, not 72.
- **Targets:** all 18 futures per observation (J = 18).
- **Predictor:** agent-local residual of width 64. Each agent sees h and only its own 3 signed bits; the others
  are masked to zero. The output is that agent's 16-dim change, with no output norm.
- **Loss:** ρ with β = 0.5, ε = 1e-4; λΔ = 2/3.
- **Prior:** 12 bits, 4 layers, `hep18_grid` CNOTs, shift b/B 8/32, r/G 12/60, 512 shots.
- **Training:** 6,000 iterations, batch 32, K = 32.
- **Readout:** per car 16 → 64 → 3, speeds 12·sigmoid(a), trapezoidal integration from the observed (s, v) at
  t = 0. Loss: position MSE/20² + speed MSE/5². 16,000 steps × 128, lr 2e-3, wd 0.01. The frozen encoder's
  future features are cached once (exact, since the encoder is frozen).
- **Metrics (F.4):**
  - joint validity and coverage over 18 futures (2 m, 1 m/s, all cars, all times);
  - TV over 19 categories;
  - minADE and ES in metres;
  - shuffled-pairing validity (10 shuffles);
  - local validity;
  - readout-only validity (the render check).

## D3IL

- **Input:** `render_segment(obs_pts)`, encoder `d = 64`.
- **Predictor:** width 128, LN output plus the bounded scale and offset head (Eq. 15).
- **Prior:** 8 bits, 4 layers, brickwork CNOTs, shift b 8, r/G 12/40, 512 shots.
- **Stage 1:** Euclidean ρ, 12,000 iterations, batch 64, K = 32.
- **Stage 2, readout:** [h; t] → 128 → 128 → 128 → 16, i.e. 8 control points of a clamped cubic B-spline,
  evaluated at u = k/24 for k = 1..24. Smooth L1 loss. 8,000 steps at 2e-3, then 2,000 at 2e-4. Batch 64,
  wd 1e-5.
- **Stage 3:** predictor and prior only, 1,000 steps at 1e-4. Each batch is 16 windows of 3 consecutive pairs.
- **Evaluation (B.4):**
  - 384 rollouts from the repeated start point, up to 16 calls each, stopping at y = 0.35 m;
  - acceptance: reaches the goal, crosses all rows, no collision with the obstacles inflated by 1 cm, stays
    within reach;
  - routes: exact expected coverage and TV over 60-path subsets (hypergeometric);
  - readout-only error on the train demos (the render check; the eval demos stay untouched).

## Order

1. Shared changes, plus the MovingMNIST regression test.
2. HEP18: readout, metrics and benchmark, test-first; then a tiny smoke run for every encoder.
3. D3IL: same steps, after `scripts/fetch_d3il.py` has produced the data.
4. Render check: readout-only diagnostics for CNN and the ViT twin. Fix σ (dataset spec §8).
