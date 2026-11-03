# Q²-JEPA — Phase 2 Design: Fair Evaluation, Quantum Parts, Image Benchmarks

Date: 2026-11-03 · Status: draft for team review · Extends [the kickstart spec](2026-10-28-q2jepa-stack-and-repo-design.md)
(stack, layout and hard rules stay as written there unless this document overrides them).

## 1. Intent

The kickstart delivered a classical end-to-end MovingMNIST pipeline and both CUDA-Q kernels. Phase 2 turns
that into the result we actually want to claim:

> HQViT (quantum attention) and the conditional Born prior (QCBM), simulated on GPUs with CUDA-Q, against
> classical counterparts that differ **only** in the quantum part — on MovingMNIST, HEP18 and D3IL, with all
> three fed to the encoder as images.

The design priority is **fairness of the comparison**. A result that is better but not attributable to the
quantum component is not a result.

## 2. Decisions since the kickstart

| Decision | Who | Overrides |
|---|---|---|
| HEP18 (Q-JEPA §5.3, App. F.3) is in scope, adapted to rendered images | team | kickstart §2 "Highway-Env deferred" |
| D3IL Avoiding (Q-JEPA §5.4, App. B) is in scope, adapted to rendered images | team | kickstart §2 non-goal |
| HQViT gradients stay inside CUDA-Q; we need the HPC part | team | resolves kickstart §9 (see §6) |
| Headline encoder comparison is HQViT vs a classical ViT twin; the CNN is an anchor to the paper | team | kickstart §4 grid |
| HQViT positional handling is changed from the paper (§5.3) | me, pending twin experiment | — |
| Reproducibility is guaranteed per platform, not across platforms (§4, rule F9) | me | clarifies kickstart §2 |

## 3. What adapting HEP18 and D3IL to images means

In the Q-JEPA paper neither task uses images. HEP18 encodes six scalars per car with an MLP; D3IL encodes a
24-point trajectory with a GRU. We render those same states as images so every encoder, HQViT included, can
consume them. Following the MovingMNIST rule:

- **Only the encoder input changes.** Data generation, splits, readout, tolerances and metrics stay in the
  paper's native state space (metres, m/s, routes).
- **Paper numbers become a reference, not a comparison.** The baselines are our own classical encoders on the
  same rendered images.
- **Sanity rows.** Each benchmark also runs once with the paper's own state-based encoder, to check our
  pipeline reproduces the paper's numbers (as the 64×64 CNN does for MovingMNIST). These are not grid rows.

## 4. Fairness protocol

These rules apply to every comparison we report. Rules marked **(hard)** are added to the hard rules in
`AGENTS.md`; breaking them invalidates a result.

- **F1 (hard) — Only the component under test changes.** An encoder comparison holds the prior, predictor,
  losses, schedule, iterations, batch, K, readout and evaluation budget fixed; a prior comparison holds the
  encoder fixed.
- **F2 (hard) — Same input** (preset: `+experiment=phase2`). Every encoder in a comparison receives the same rendered images at the same
  `encoder.res`, with the same `encoder.input_norm`. For any comparison involving HQViT this is `l2`
  (unit-RMS frames): amplitude encoding discards overall brightness, so no classical encoder may see it.
- **F3 — Same frame fusion.** HQViT encodes each frame separately and fuses after pooling. The twin does the
  same. The CNN stacks frames as channels (early fusion); this difference is stated next to every CNN row.
- **F4 (hard) — HQViT claims are made against the twin.** The twin (`encoder=vit`) shares HQViT's patching,
  block count, FFN, positional handling, pooling and fusion head; only `Block.attend` differs.
- **F5 — Equal tuning budget, dev split only.** Every encoder and prior gets the same number of tuning runs,
  selected on the dev split. The test split is touched once per final checkpoint.
- **F6 — Paired seeds, at least 3.** Seeds 0–2 for every row; compare per seed as well as mean ± sample s.d.
  If the gap between two rows is within one s.d., add seeds 3–4 before drawing a conclusion. (Measured
  spread for the CNN at 16×16: ±3.1 coverage points, ±0.029 TV over 3 seeds.)
- **F7 — Same steps, not same wall-clock.** Iteration counts follow Q-JEPA Table 5. Wall-clock, circuit
  evaluations and shots are reported separately as the HPC result.
- **F8 — Report sizes.** Every results row lists trainable parameters split into quantum and classical.
- **F9 (hard) — One platform per comparison.** The same overrides, seed and SHA give different numbers on
  different hardware (seed 0, 16×16 CNN: 95.8% validity on the Mac, 95.2% on Linux). All rows of one
  comparison table run on the same platform (e.g. all on Colab T4), which is recorded in the row.
- **F10 — Readout-only diagnostic.** Every row also reports metrics with encoded real targets in place of
  predictions, separating "the encoder loses information" from "the world model predicts badly".
- **F11 — State simulation mode.** Training uses exact factored HQViT attention and finite-shot QCBM sampling
  as specified in §6–7. Any shot-based component is named in the row.

## 5. Encoders

All three implement `forward(x: (B, 3, res, res)) -> (B, 64)` and are built by `encoder.build_encoder`.

### 5.1 CNN (done)
Q-JEPA App. A.2. Frames stacked as channels. Used with `input_norm=l2` in all phase-2 grid rows.

### 5.2 Classical ViT twin (`encoder=vit`, built 2026-11-03)
`src/q2jepa/encoder/vit.py`. 4×4 patches (16 tokens of dimension 16 at 16×16), `depth=3` single-head post-norm
blocks, FFN hidden 16 (HQViT §5), each frame encoded separately, then pooled and fused by the CNN's head widths
(·→96→64, LayerNorm). Patch layout matches `quantum/attention.py` exactly, so HQViT can reuse `to_patches`.

### 5.3 Positional information (deviation from HQViT)
HQViT's ansatz is `I_idx ⊗ W(θ)`: it never acts on the patch-index qubits. Attention scores therefore depend
only on patch contents, and after mean pooling each frame's embedding is invariant to permuting its patches.
Regression-tested for both the factored and the full 17-qubit kernels (`test_attention_equivalence.py`) and for the twin (`test_vit.py`).
For classification this is harmless; for motion it removes the information the target encoder needs.

Options, applied identically to HQViT and the twin:

| `encoder.pool` / `encoder.pos_emb` | Effect | Keeps factored path |
|---|---|---|
| `mean` / `false` | paper-faithful, position-blind | yes |
| `flatten` / `false` | concatenate tokens before the head | yes |
| `mean` / `true` | learned positional embedding added to tokens before block 1 | yes |
| non-identity ansatz on index qubits | quantum positional mixing | **no** — rejected |

**Decision rule:** the twin is trained with all settings on MovingMNIST (seed 0, then 3 seeds for the
two best). We adopt the best setting for both encoders and also report the paper-faithful one as an
ablation. Reportable reruns go to `docs/results/` with overrides, seed, SHA, platform and parameter counts.

**First exploratory runs** (2026-11-03, seed 0, 16×16, `input_norm=l2`, Softmax-Categorical, uncommitted tree
on top of 4b7fe38, Linux CPU; not reportable):

| Encoder | Params | Cov. | Valid | TV | minADE | ES |
|---|---|---|---|---|---|---|
| CNN | 70,960 | 87.0 | 93.1 | 0.207 | 1.30 | 9.67 |
| twin, mean / no pos (HQViT-faithful) | 15,312 | 19.8 | 22.9 | 0.817 | 4.30 | 11.35 |
| twin, flatten | 84,432 | 36.9 | 41.4 | 0.652 | 3.39 | 10.57 |
| twin, mean / learned pos (zero init) | 15,568 | 18.8 | 21.0 | 0.831 | 4.39 | 11.46 |

The position-blind design collapses as predicted. A zero-initialized learned embedding barely moves: the
encoder learning rate drops to 1e-5 after iteration 200. Flattening helps but the twin stays far below the
CNN. Next: a fixed 2-D sinusoidal embedding (no learning needed), readout-only diagnostics to locate the loss
of information, and a twin learning-rate check under F5's equal tuning budget.

### 5.4 HQViT (`encoder=hqvit`, not built)
Reuses the twin's skeleton; replaces `Block.attend` with:
1. amplitude-encode the block's input tokens (whole-image normalization, HQViT Eq. 5);
2. Q and K patch states from `quantum/attention.py` (factored mode), A_ij = w_i w_j |⟨q̂_i|k̂_j⟩|²;
3. V from the V-register ansatz, read out as per-token measurement probabilities;
4. softmax(A)·V, then the twin's residual, FFN and LayerNorm.
Between blocks, token outputs are re-encoded into amplitudes (HQViT §6 notes this conversion cost).
The full 17-qubit mode stays an equivalence test and benchmark only (kickstart §6).

## 6. HQViT gradients through CUDA-Q

The circuit parameters θQ, θK, θV are opaque to PyTorch, so `quantum/bridge.py` supplies their gradients
through a `torch.autograd.Function`.

| Option | Cost per step | Status |
|---|---|---|
| A. Parameter shift (±π/2 per angle) | ~2 × (angles per circuit) × circuits extra runs, ~100–170× forward | fallback only |
| B. Exact gradients from CUDA-Q statevectors (`cudaq.get_state` of the patch circuits, derivative states per angle) | a few forward passes | **chosen** |
| C. CUDA-Q builds the ansatz matrix once per step, torch applies it to all patches | ~57 small circuits per step, batch-independent | **rejected**: per-patch evolution would leave CUDA-Q and gut the HPC story |

Gate before building HQViT: time one forward pass (and one option-B backward pass) for a batch of 64 at
16×16 on a Colab T4, against the expected budget of 4,000 iterations × 2 branches × 3 frames. If it doesn't
fit, the fallbacks are, in order: batching patch circuits inside one CUDA-Q call (`cudaq.get_state` over
argument lists, or `nvidia-mqpu`), fewer blocks, then a smaller batch — never option C.

Equivalence test required: option-B gradients match finite differences and option-A parameter shift on a
small case, on both `qpp-cpu` and `nvidia`.

## 7. QCBM prior

Following Q-JEPA §3, App. A.4 and App. C.

- **Module** `prior/qcbm.py` with the classical-prior interface `sample(h, k)` / `log_prob(h, z)`.
  Conditioner `g_η`: LayerNorm, d→48, SiLU, 48→(L+1)·n with zero-initialized output; angles
  α = g_η(h) + ϑ (offsets ϑ trainable).
- **Circuits** from `quantum/born.py`: MovingMNIST n=3, L=2 (CNOT (0,1), (1,2)); HEP18 n=12, L=4 (4×3 grid
  layout, App. C); D3IL n=8, L=4 brickwork. Product-QCBM drops the CNOTs; Frozen-QCBM never updates.
- **Gradient modes** (`quantum.grad`): `ideal` (score-function through `born_probs`, reference only),
  `shift` (parameter shift, App. A.4: b of B observations, r of G coordinates, shots per shifted circuit),
  `shift_noise` (plus 0.5% readout flips). Grid rows use `shift_noise`, as in the paper's tables.
- **Training loop.** The prior update is computed separately from backprop (App. A.4 step 4) and written into
  the prior's `.grad` before the shared clip-and-step. `train.py` gets a prior hook; classical priors keep
  their current score-function path.
- **Configs** `conf/prior/{product_qcbm,entangled_qcbm,frozen_qcbm}.yaml`; new `QuantumCfg` keys `grad`,
  `shots_eval`.
- **Evaluation noise.** QCBM rows also apply the 0.5% flips at evaluation (App. D).

## 8. Evaluation (paper-exact)

### 8.1 Common
- `q2jepa.evaluate.evaluate(cfg, out_dir, sampler=None) -> dict` loads `ckpt.pt` + `readout.pt` and computes
  metrics; `train.py` calls it at the end. `scripts/eval_hardware.py` passes an IBM sampler.
- Every run reports test metrics, readout-only metrics (F10), dev metrics, parameter counts (F8), wall-clock,
  circuit evaluations and shots.
- `docs/results/*.md` tables report mean ± sample s.d. over seeds, plus the per-seed values.

### 8.2 MovingMNIST (Q-JEPA App. F.2, F.4)
Already matching: data construction, 80 test observations × 128 draws including repeated codes, coverage,
validity (4 px endpoint), TV with invalid category, probability-weighted minADE, trajectory ES.
To add: best-sample **Dice** weighted by direction probability; readout-only; dev split (80 observations,
already generated) for selection.

### 8.3 HEP18 (Q-JEPA §5.3, App. F.3, F.4)
- **Data** `data/hep18.py`: 448 observations (256 train / 64 val / 128 test), 18 equally likely joint futures
  per observation (priority axis × 3 schedules per priority car; yielding cars brake), constructed from the
  schedules in App. F.3 on HighwayEnv's road geometry. All 18 futures are training targets (18 target views
  per observation in a batch).
- **Rendering (new).** One image clip **per car**, rendered top-down with only that car drawn: observation
  frames at −2, −1, 0 s, target frames at 3, 6, 9 s. This preserves the paper's per-agent encoder exactly (each
  car's encoder sees only its own motion) and yields 4 × 16-d agent representations concatenated to d=64.
  Cars are drawn as anti-aliased Gaussian blobs so sub-pixel position survives downsampling. Open: crop, blob
  width and `encoder.res` (16 vs 32); decided by the readout-only diagnostic of the CNN.
- **Model differences from MovingMNIST** (paper, unchanged): agent-local bits (3 per car, other cars' bits
  masked), residual predictor without output norm, smoothed power distance ρ (β=0.5, ε=1e-4) for training,
  change regularization λΔ R(t−h) with λΔ = 2/3, 6,000 iterations, batch 32, readout 16,000 × 128, b/B 8/32,
  r/G 12/60, 512 shots.
- **Readout** `readout/hep18.py`: per-car 16→64→3 MLP giving speeds bounded by 12·sigmoid, integrated with the
  trapezoidal rule from the observed state.
- **Metrics:** joint validity (every car ≤ 2 m and ≤ 1 m/s at all three times), coverage of 18, TV with invalid
  category, uniform-weighted minADE, ES in metres, shuffle-the-cars coordination check (10 permutations),
  local validity.
- **Encoder width.** Each car's encoder outputs 16-d instead of 64-d (`encoder.d=16`); the per-car clip is
  encoded with shared weights.
- **Sanity row:** paper's scalar MLP encoder, Entangled-QCBM and Softmax-Categorical, vs Table 3.

### 8.4 D3IL Avoiding (Q-JEPA §5.4, App. B)
- **Data** `data/d3il.py`: the 96 public demonstrations, three route-stratified folds of 64/32, 8 cm
  arc-length segments resampled to 24 points, first observation interpolated from frames 4–8, coordinates
  normalized as (xy − (0.5, −0.28)) / 0.4. To check: download source and licence; whether the Avoiding task
  ships camera observations we could use instead of rasterizing.
- **Rendering (new).** A 24-point segment becomes a 3-frame clip, points 1–8, 9–16, 17–24, drawn as
  anti-aliased polylines in fixed workspace coordinates (obstacle positions matter, so no agent-centric crop).
  Splitting into frames keeps the direction of travel, which a single polyline image would lose. The
  stationary start renders as three identical dots. Open: `encoder.res` — 16×16 is ~5 cm/pixel against 8 cm
  segments, so 32×32 is likely needed; decided by the readout-only diagnostic.
- **Recursive evaluation.** Decode a segment, re-render it, re-encode, repeat until y crosses 0.35 m or 16
  calls. Rendering is therefore in the evaluation loop and must be deterministic and batched.
- **Model and training** (paper): 8 bits, entangled 4-layer brickwork QCBM (40 rotations, 14 CNOTs), width-128
  predictor with the bounded scale/offset head (Eq. 15), 12,000 updates, readout spline MLP (8 control
  points, clamped cubic B-spline) with Smooth-L1 for 8,000 + 2,000 steps, then 1,000 predictor/prior steps on
  real segment pairs. Parameter shift: 8 observations, 12 of 40 angles, 512 shots, 0.5% flips.
- **Metrics:** 384 rollouts per checkpoint from the shared stationary start; geometric acceptance (goal
  reached, all rows crossed, continuous collision check with 1 cm inflation, reach ≤ 1.0913 m); route
  coverage of 24 and TV against uniform, as exact hypergeometric expectations over 60-path subsets;
  3 folds × 3 seeds, folds averaged within seed, then mean ± s.d. of seed means.
- **Sanity row:** paper's GRU + kinematics encoder on raw points, vs Table 4.

## 9. Experiment grid

Per benchmark: encoder {CNN, twin, HQViT} × prior {Softmax-Categorical, Entangled-QCBM} × 3 seeds.

| Benchmark | Runs | Notes |
|---|---|---|
| MovingMNIST | 18 | + positional ablation (twin, HQViT) + Product/Frozen-QCBM rows if budget allows |
| HEP18 | 18 | + Product-QCBM row (it's where entanglement matters, Table 3) |
| D3IL | 54 models (× 3 folds) | priority: twin and HQViT with Entangled-QCBM first |
| Sanity rows | ~4 | paper encoders on native inputs; one seed |

The GPU budget is still open (kickstart §8). Priority order if it's tight: MovingMNIST grid → HEP18 HQViT vs
twin with Entangled-QCBM → D3IL HQViT vs twin with Entangled-QCBM → everything else.

## 10. Build order and gates

1. **Evaluation refactor** (§8.1, §8.2): `evaluate()`, Dice, readout-only, dev metrics, results-table helper.
2. **Twin positional experiment** (§5.3) — running; decides `pool`/`pos_emb`.
3. **QCBM prior** (§7) with CNN; first quantum rows on CPU (3 qubits).
4. **Gate: T4 timing** of HQViT forward/backward (§6).
5. **HQViT** (§5.4, §6), equivalence tests on `qpp-cpu` and `nvidia`.
6. **MovingMNIST grid** on one platform (F9).
7. **IBM inference** (kickstart §6) on the trained MovingMNIST Born prior.
8. **HEP18**: data, renderer, readout, metrics; **gate:** CNN readout-only diagnostic decides `res`/crop.
9. **D3IL**: data, renderer, recursive evaluation, metrics; same gate.

Steps 1–3 and the HEP18/D3IL data modules need no GPU and can proceed in parallel.

## 11. Still open

- **GPU budget** (team) — sizes the grid and decides whether `encoder.res=32` is affordable for HEP18/D3IL.
- **IBM access** (team) — backend and minutes, needed at step 7.
- **D3IL data source** — download location, licence, and whether camera observations exist.
- **HEP18/D3IL rendering parameters** — crop, blob/line width, resolution; decided by gates in §10.
- **Twin positional result** — decides §5.3.
