# Q²-JEPA Repo Kickstart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the Q²-JEPA repo through a classical end-to-end MovingMNIST run, with both CUDA-Q kernels (Born prior sampling, swap-test attention) built and tested, plus a GPU benchmark script.

**Architecture:** One `uv`-managed package `q2jepa` under `src/`, driven by Hydra configs whose values come from Q-JEPA Table 5. All `cudaq` code sits in `q2jepa.quantum`. The classical pipeline is CNN encoder → Softmax-Categorical/Mixture prior → shared predictor → energy score + VISReg → frozen translation readout → paper metrics, and it must reproduce the paper's classical numbers before any quantum module enters training.

**Tech Stack:** Python 3.14, uv, PyTorch, torchvision, cudaq ~=0.16.0, hydra-core 1.3, wandb, numpy, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-28-q2jepa-stack-and-repo-design.md`. Paper values come from `docs/qjepa.pdf` (Q-JEPA: §3, App. A, C, F.2, F.4) and `docs/hqvit.pdf` (§4.2, Eqs. 11–22).

## Global Constraints

- `requires-python = ">=3.12"` (Colab is 3.12), `.python-version` = `3.14` for local dev. CUDA-Q comes from `cuda-quantum-cu13==0.16.0` (the cu12 swap happens on Colab only). Floors are loose: `torch>=2.5`, `torchvision>=0.20`, `numpy>=1.26`, `wandb>=0.18`.
- No syntax newer than 3.12 (e.g. no PEP 695 `type` aliases or generics syntax). `uv run --python 3.12 pytest` must pass as well as 3.14.
- In `src/`, `import cudaq` appears only under `src/q2jepa/quantum/`. Tests may import it directly. No `qiskit` in this plan.
- **Bit/qubit convention (verified on cudaq 0.16):** the state-vector index is little-endian (qubit 0 = least-significant bit), and sample bitstrings print qubit 0 first. Across the whole repo, `z[..., j]` is qubit/bit `j`, and the integer code index is `Σ_j z_j·2^j`.
- Only the encoder input is downsampled (`encoder.res` ∈ {16, 32, 64}). Data, readout and metrics stay at 64×64.
- Table 5 (MovingMNIST) values: d = 64, predictor width 96, Hp = 48, n = 3 bits, L = 2, iters 4000, batch 64, K = 32 train / 128 eval, warm-up 200/300, encoder LR 3e-4 → 1e-5, predictor/prior LR 1e-3, AdamW wd 1e-5, per-group grad clip 5, λ_reg = 4/3, VISReg J = 128 / λc = 1 / λs = 10 / shape ramp over 100 iters, 256 shots per shifted circuit, b/B = 8/64, r/G = 9/9, readout flip 0.005, readout 4000 iters / batch 64 / LR 2e-3 / wd 0.01.
- HQViT ansatz `W(θ)` on m patch qubits: D units of (RX·RY·RX on each qubit, then a CNOT ring j→(j+1) mod m), then a final RY layer, giving `3·m·D + m` parameters (28 for m = 4, D = 2).
- Torch device `trainer.device: auto` means `cuda` if available, else `cpu` (MPS is not used: grid_sample determinism).

## Review Focus

1. **Empty patches.** At 16×16, most 4×4 patches of a MovingMNIST frame are all-zero, so amplitude encoding divides by zero. Expect zero weight and no NaN. Test lives in Task 4.
2. **Code bit order across priors.** If classical index→bits disagrees with cudaq's little-endian bits, codes silently mean different things in different priors. Test lives in Task 7.
3. **All-black image into the attention kernel** (‖X‖ = 0). Expect a clear `ValueError`, not NaNs. Test lives in Task 4.
4. **Rendering vs metric convention.** Future frames, readout warping and the true trajectories must all agree that +dx is right and +dy is down. Test lives in Task 5.
5. **Reproducibility.** The same seed and config must give identical `metrics.json` on CPU. Test lives in Task 10.

---

### Task 1: Repo bootstrap

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/q2jepa/__init__.py`, `src/q2jepa/quantum/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: `q2jepa.quantum.set_target(name: str) -> None` (wraps `cudaq.set_target`). `tests/conftest.py` calls it once per session with `os.environ.get("Q2JEPA_CUDAQ_TARGET", "qpp-cpu")`.

- [ ] **Step 1: Init git and uv**

```bash
cd /Users/paulhondola/Developer/Q2-JEPA
git init -b main
uv init --lib --name q2jepa --python 3.14
uv add "torch>=2.5" "torchvision>=0.20" "cuda-quantum-cu13==0.16.0" "hydra-core~=1.3" "wandb>=0.18" "numpy>=1.26"
uv add --dev pytest ruff
```

Edit `pyproject.toml` so that `requires-python = ">=3.12"`, and add `[tool.pytest.ini_options] pythonpath = ["."]` (so tests can import `scripts.*`). `.gitignore` must contain `data/`, `outputs/`, `multirun/`, `wandb/`, `.venv/`, `__pycache__/`, `.DS_Store`, `.remember/`.

- [ ] **Step 2: Write the failing smoke test** in `tests/test_smoke.py`

```python
import cudaq
from q2jepa.quantum import set_target

def test_cudaq_target_selected():
    set_target("qpp-cpu")
    assert cudaq.get_target().name == "qpp-cpu"
```

- [ ] **Step 3: Run it.** `uv run pytest tests/test_smoke.py -v`. Expected: FAIL (`set_target` not defined).
- [ ] **Step 4: Implement `set_target`** and `tests/conftest.py` (a session-scoped autouse fixture that reads the env var).
- [ ] **Step 5: Run it.** `uv run pytest -v`. Expected: PASS.
- [ ] **Step 6: Commit on main, then branch**

```bash
git add -A && git commit -m "chore: bootstrap uv project with cudaq, torch, hydra"
git switch -c kickstart
```

---

### Task 2: Config schema + Hydra skeleton

**Files:**
- Create: `src/q2jepa/config.py`, `conf/config.yaml`, `conf/encoder/cnn.yaml`, `conf/prior/softmax_categorical.yaml`, `conf/prior/mixture.yaml`, `conf/dataset/mmnist.yaml`, `conf/quantum/cpu.yaml`, `conf/quantum/gpu.yaml`, `conf/trainer/default.yaml`, `src/q2jepa/train.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces dataclasses in `q2jepa.config`, with field names fixed as below. Importing the module registers `Config` in `ConfigStore` as `base_config`.
  - `EncoderCfg(name="cnn", res=16, d=64)`
  - `PriorCfg(name="softmax_categorical", n_bits=3, hidden=48, components=2, layers=2)`
  - `PredictorCfg(width=96)`
  - `QuantumCfg(target="qpp-cpu", shots=256, readout_flip=0.005, shift_obs=8, shift_coords=9)`
  - `TrainerCfg(iters=4000, batch=64, k_train=32, warmup_repr=200, warmup_pred=300, lr_enc=3e-4, lr_enc_later=1e-5, lr_pred=1e-3, lr_prior=1e-3, weight_decay=1e-5, clip=5.0, lambda_reg=4/3, device="auto")`
  - `VISRegCfg(n_proj=128, w_center=1.0, w_scale=10.0, shape_ramp=100)`
  - `ReadoutCfg(iters=4000, batch=64, lr=2e-3, weight_decay=0.01, clip=5.0)`
  - `EvalCfg(n_obs=80, k_eval=128)`
  - `DatasetCfg(name="mmnist", root="data", n_train=800, n_dev=80, n_test=80)`
  - `WandbCfg(mode="offline", project="q2jepa")`
  - `Config(seed=0, encoder, prior, predictor, quantum, trainer, visreg, readout, eval, dataset, wandb)`
- `conf/config.yaml` defaults list: `base_config`, `encoder: cnn`, `prior: softmax_categorical`, `dataset: mmnist`, `quantum: cpu`, `trainer: default`, `_self_`. `quantum/gpu.yaml` sets `target: nvidia`. `prior/mixture.yaml` sets `name: mixture`.
- `q2jepa.train.main` is `@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")`. In this task it only prints `OmegaConf.to_yaml(cfg)`.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
import pytest
from hydra import compose, initialize_config_dir
import q2jepa.config  # noqa: F401  registers schema

CONF = str(Path(__file__).resolve().parents[1] / "conf")

def _cfg(overrides=()):
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        return compose("config", overrides=list(overrides))

def test_table5_defaults():
    c = _cfg()
    assert (c.encoder.d, c.predictor.width, c.prior.hidden, c.prior.n_bits, c.prior.layers) == (64, 96, 48, 3, 2)
    t = c.trainer
    assert (t.iters, t.batch, t.k_train, t.warmup_repr, t.warmup_pred) == (4000, 64, 32, 200, 300)
    assert (t.lr_enc, t.lr_enc_later, t.lr_pred, t.lr_prior, t.weight_decay, t.clip) == (3e-4, 1e-5, 1e-3, 1e-3, 1e-5, 5.0)
    assert t.lambda_reg == pytest.approx(4 / 3)
    assert (c.visreg.n_proj, c.visreg.w_center, c.visreg.w_scale, c.visreg.shape_ramp) == (128, 1.0, 10.0, 100)
    assert (c.quantum.shots, c.quantum.readout_flip, c.quantum.shift_obs, c.quantum.shift_coords) == (256, 0.005, 8, 9)
    assert (c.readout.iters, c.readout.batch, c.readout.lr, c.readout.weight_decay) == (4000, 64, 2e-3, 0.01)
    assert (c.eval.n_obs, c.eval.k_eval) == (80, 128)

def test_group_switches():
    assert _cfg(["quantum=gpu"]).quantum.target == "nvidia"
    assert _cfg(["prior=mixture"]).prior.name == "mixture"

def test_typo_rejected():
    with pytest.raises(Exception):
        _cfg(["trainer.itters=5"])
```

- [ ] **Step 2: Run.** `uv run pytest tests/test_config.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `config.py`, the YAML files and the print-only `train.py`.
- [ ] **Step 4: Run.** `uv run pytest tests/test_config.py -v` gives PASS, and `uv run python -m q2jepa.train quantum=gpu` prints YAML containing `target: nvidia`.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: hydra config schema with Q-JEPA Table 5 defaults"`

---

### Task 3: Born-machine kernel

**Files:**
- Create: `src/q2jepa/quantum/born.py`
- Test: `tests/test_born.py`

**Interfaces:**
- Produces:
  - `MMNIST_CNOT_LAYERS: list[list[tuple[int, int]]] = [[(0, 1)], [(1, 2)]]` (Q-JEPA App. C; pairs are (control, target)).
  - `born_probs(angles: np.ndarray, cnot_layers) -> np.ndarray`: `angles` has shape `(L+1, n)`. Rows 0..L−1 are body RY angles and row L holds the terminal RY angles φ. Returns shape `(2**n,)`, indexed little-endian, computed from `cudaq.get_state`.
  - `sample_codes(angles: np.ndarray, cnot_layers, shots: int, rng: np.random.Generator) -> np.ndarray`: `angles` has shape `(B, L+1, n)`. Returns uint8 `(B, shots, n)` in shuffled order.
  - `apply_readout_flips(bits: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray`: XOR with Bernoulli(p) per bit, applied classically after sampling. This equals an independent readout bit-flip channel.
- Circuit (Q-JEPA Eq. 4): for ℓ in 0..L−1, apply `ry(angles[ℓ, j])` on every qubit j, then the CNOTs of `cnot_layers[ℓ]`. Then apply `ry(angles[L, j])` and measure. The kernel takes flat `list[float]`/`list[int]` args (n, flattened angles, flattened CNOT pairs, pairs-per-layer). `cudaq.sample` broadcasts over a 2-D float array for a `list[float]` argument; `get_state` does not, so loop for `born_probs`.

- [ ] **Step 1: Write the failing tests.** `_np_born` is a tiny numpy statevector reference that uses the same little-endian convention.

```python
import numpy as np
from q2jepa.quantum.born import MMNIST_CNOT_LAYERS, born_probs, sample_codes, apply_readout_flips

def _np_born(angles, layers):
    n = angles.shape[1]; psi = np.zeros(2**n); psi[0] = 1.0
    def ry(psi, j, a):
        c, s = np.cos(a / 2), np.sin(a / 2); out = psi.copy()
        for i in range(2**n):
            if not (i >> j) & 1:
                k = i | (1 << j); out[i] = c * psi[i] - s * psi[k]; out[k] = s * psi[i] + c * psi[k]
        return out
    def cx(psi, ctl, tgt):
        out = psi.copy()
        for i in range(2**n):
            if (i >> ctl) & 1: out[i] = psi[i ^ (1 << tgt)]
        return out
    for l, pairs in enumerate(layers):
        for j in range(n): psi = ry(psi, j, angles[l, j])
        for c, t in pairs: psi = cx(psi, c, t)
    for j in range(n): psi = ry(psi, j, angles[-1, j])
    return psi**2

def test_entangled_matches_reference():
    a = np.random.default_rng(0).uniform(-np.pi, np.pi, (3, 3))
    np.testing.assert_allclose(born_probs(a, MMNIST_CNOT_LAYERS), _np_born(a, MMNIST_CNOT_LAYERS), atol=1e-6)

def test_product_marginals():
    a = np.random.default_rng(1).uniform(-np.pi, np.pi, (3, 3))
    p = born_probs(a, [[], []])
    marg = [p[[i for i in range(8) if (i >> j) & 1]].sum() for j in range(3)]
    np.testing.assert_allclose(marg, np.sin(a.sum(0) / 2) ** 2, atol=1e-6)

def test_bit_order_qubit0_is_z0():
    a = np.zeros((3, 3)); a[-1, 0] = np.pi
    z = sample_codes(a[None], MMNIST_CNOT_LAYERS, 64, np.random.default_rng(0))
    assert z.shape == (1, 64, 3) and (z[0] == [1, 0, 0]).all()

def test_sample_frequencies_match_probs():
    rng = np.random.default_rng(2); a = rng.uniform(-np.pi, np.pi, (2, 3, 3))
    z = sample_codes(a, MMNIST_CNOT_LAYERS, 20000, rng)
    for b in range(2):
        idx = (z[b] * [1, 2, 4]).sum(-1)
        freq = np.bincount(idx, minlength=8) / 20000
        assert 0.5 * np.abs(freq - born_probs(a[b], MMNIST_CNOT_LAYERS)).sum() < 0.02

def test_readout_flips():
    rng = np.random.default_rng(0); bits = rng.integers(0, 2, (4, 1000, 3)).astype(np.uint8)
    assert (apply_readout_flips(bits, 0.0, rng) == bits).all()
    assert (apply_readout_flips(bits, 1.0, rng) == 1 - bits).all()
    assert abs((apply_readout_flips(bits, 0.005, rng) != bits).mean() - 0.005) < 0.002
```

- [ ] **Step 2: Run.** `uv run pytest tests/test_born.py -v`. Expected: FAIL (import error).
- [ ] **Step 3: Implement `born.py`.**
- [ ] **Step 4: Run.** `uv run pytest tests/test_born.py -v`. Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: cudaq conditional Born-machine sampling kernel"`

---

### Task 4: Swap-test attention kernels (factored 4q + full 2n+1 q)

**Files:**
- Create: `src/q2jepa/quantum/attention.py`
- Test: `tests/test_attention_equivalence.py`

**Interfaces:**
- Produces:
  - `image_to_patches(img: np.ndarray, p: int = 4) -> np.ndarray`: `(H, W)` → `(T, p*p)`, with patches ordered row-major over the grid and pixels row-major within each patch.
  - `n_ansatz_params(m: int, depth: int) -> int` = `3*m*depth + m`.
  - `patch_states(patches: np.ndarray, theta: np.ndarray, depth: int = 2) -> tuple[np.ndarray, np.ndarray]` returns `(w, states)`. `w[i] = ‖x_i‖²/‖X‖²` with shape `(T,)`. `states[i] = W(θ)|x̂_i⟩` with shape `(T, d)`, complex, from `cudaq.get_state` of an m-qubit kernel (`cudaq.qvector(amps)` init plus ansatz). A patch with `w[i] == 0` gets state `|0…0⟩` and is never sent to cudaq. If ‖X‖ = 0, raise `ValueError`.
  - `factored_acm(patches, theta_q, theta_k, depth=2) -> np.ndarray` `(T, T)`: `ACM_ij = w_i w_j |⟨q̂_i|k̂_j⟩|²`, which is HQViT Eq. 22 with unnormalized qᵢ.
  - `full_swap_test_probs(patches, theta_q, theta_k, depth=2) -> np.ndarray` `(T, T)`: `Pr(i, j, anc=0)` from one `(2n+1)`-qubit `cudaq.get_state`.
- **Full-circuit qubit layout:** RQ = qubits `0..n−1` (patch qubits `0..m−1`, index qubits `m..n−1`), RK = qubits `n..2n−1` (patch qubits `n..n+m−1`), ancilla = qubit `2n`. Both registers are amplitude-encoded with the whole image `X/‖X‖` (flattened as `patches.reshape(-1)`, so index `= i·d + l`, which puts patch qubits low). Apply `W(θ_q)` on RQ's patch qubits and `W(θ_k)` on RK's patch qubits. Then H(anc), CSWAP(anc; RQ patch qubit r ↔ RK patch qubit r) for r in 0..m−1, H(anc). Marginalize with:

```python
P = np.abs(state) ** 2
P = P.reshape(2, T, d, T, d)       # (anc, iK, lK, iQ, lQ) under little-endian indexing
pr = P[0].sum(axis=(1, 3)).T       # -> [iQ, jK]
```

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, pytest
from q2jepa.quantum.attention import image_to_patches, n_ansatz_params, factored_acm, full_swap_test_probs

def _setup(seed, img):
    rng = np.random.default_rng(seed); k = n_ansatz_params(4, 2)
    return image_to_patches(img, 4), rng.uniform(-np.pi, np.pi, k), rng.uniform(-np.pi, np.pi, k)

def test_param_count():
    assert n_ansatz_params(4, 2) == 28

def test_full_equals_factored_random_image():
    img = np.random.default_rng(0).random((16, 16))
    x, tq, tk = _setup(1, img)
    w = (np.linalg.norm(x, axis=1) ** 2) / (np.linalg.norm(x) ** 2)
    np.testing.assert_allclose(full_swap_test_probs(x, tq, tk), 0.5 * (np.outer(w, w) + factored_acm(x, tq, tk)), atol=1e-6)

def test_empty_patches_no_nan():
    img = np.zeros((16, 16)); img[5:11, 6:10] = np.random.default_rng(2).random((6, 4))
    x, tq, tk = _setup(3, img)
    f = factored_acm(x, tq, tk)
    assert np.isfinite(f).all()
    empty = np.linalg.norm(x, axis=1) == 0
    assert empty.sum() >= 8 and (f[empty] == 0).all() and (f[:, empty] == 0).all()
    w = (np.linalg.norm(x, axis=1) ** 2) / (np.linalg.norm(x) ** 2)
    np.testing.assert_allclose(full_swap_test_probs(x, tq, tk), 0.5 * (np.outer(w, w) + f), atol=1e-6)

def test_black_image_raises():
    x, tq, tk = _setup(0, np.zeros((16, 16)))
    with pytest.raises(ValueError):
        factored_acm(x, tq, tk)
```

- [ ] **Step 2: Run.** `uv run pytest tests/test_attention_equivalence.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `attention.py`.** The ansatz is a cudaq kernel helper taking `(qubits, theta: list[float], depth: int)` and is reused by both entry points.
- [ ] **Step 4: Run.** Expected: PASS (17-qubit statevector, a few seconds on CPU).
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: HQViT swap-test attention kernels, factored and full, with equivalence test"`

---

### Task 5: Stochastic MovingMNIST

**Files:**
- Create: `src/q2jepa/data/__init__.py`, `src/q2jepa/data/mmnist.py`
- Test: `tests/test_mmnist.py`

**Interfaces:**
- Produces:
  - `ANGLES_DEG = np.arange(8) * 45`. Direction k means motion `(cos θ_k, sin θ_k)` in image coordinates (+x right, +y down; *planner's call*).
  - `DIRECTION_PROBS = np.array([0.15, 0.10] * 4)`, where even k is cardinal. `RECORDS_PER_DIRECTION = [3, 2] * 4`.
  - `true_trajectories() -> np.ndarray` with shape `(8, 3, 2)`: `[k, f-1] = 4·f·(cos θ_k, sin θ_k)` for f = 1..3. These are displacements from the last observed centre.
  - `translate(images: torch.Tensor, disp: torch.Tensor) -> torch.Tensor`: images `(..., H, W)`, disp `(..., 2)` in pixels (dx, dy). Uses `F.affine_grid`/`F.grid_sample`, bilinear, zero padding, `align_corners=True`. The readout reuses it.
  - `StochasticMovingMNIST(split: Literal["train", "dev", "test"], root: str = "data", n_obs: int | None = None, seed: int = 0)`, a `torch.utils.data.Dataset`. Items are dicts: `obs` `(3, 64, 64)` float in [0, 1], `fut` `(3, 64, 64)`, `direction` int, `obs_idx` int. The train split has 20 records per observation (3 per cardinal, 2 per diagonal, permuted). Dev and test have 8 records per observation (one per direction). Defaults: n_obs = 800/80/80. Train and dev draw disjoint MNIST-train indices; test draws from MNIST-test. Also provides `unique_obs() -> torch.Tensor` with shape `(n_obs, 3, 64, 64)`.
- Rendering (Q-JEPA F.2): paste the 28×28 digit at canvas offset (18, 18) so its centre is (32, 32). Observation frame centres are x = 18, 21, 24 at y ~ U{24..40}. Each frame is `translate(canvas, (cx−32, cy−32))`. Future frame f is `translate(last_obs_frame, true_trajectories()[k, f-1])`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, torch
from q2jepa.data.mmnist import DIRECTION_PROBS, true_trajectories, translate, StochasticMovingMNIST

def test_probs_and_trajectories():
    assert np.isclose(DIRECTION_PROBS.sum(), 1.0)
    t = true_trajectories()
    np.testing.assert_allclose(t[0, 2], [12, 0], atol=1e-9)
    np.testing.assert_allclose(t[2, 0], [0, 4], atol=1e-9)

def test_translate_moves_centroid():
    img = torch.zeros(64, 64); img[30, 20] = 1.0
    out = translate(img, torch.tensor([3.0, -2.5]))
    ys, xs = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    assert abs((out * xs).sum() / out.sum() - 23.0) < 0.05 and abs((out * ys).sum() / out.sum() - 27.5) < 0.05

def test_train_records_and_render_convention():
    ds = StochasticMovingMNIST("train", n_obs=2)
    assert len(ds) == 40
    dirs = [ds[i]["direction"] for i in range(20) if ds[i]["obs_idx"] == ds[0]["obs_idx"]]
    assert np.bincount(dirs, minlength=8).tolist() == [3, 2] * 4
    item = next(ds[i] for i in range(40) if ds[i]["direction"] == 0)
    expected = translate(item["obs"][2], torch.tensor([4.0, 0.0]))
    assert torch.allclose(item["fut"][0], expected, atol=1e-5)
    assert item["obs"].min() >= 0 and item["obs"].max() <= 1

def test_eval_split_and_disjointness():
    te = StochasticMovingMNIST("test", n_obs=3)
    assert len(te) == 24 and te.unique_obs().shape == (3, 3, 64, 64)
    tr, dv = StochasticMovingMNIST("train", n_obs=800), StochasticMovingMNIST("dev", n_obs=80)
    assert set(tr.mnist_indices).isdisjoint(dv.mnist_indices)
```

(`mnist_indices` is a public list attribute giving the MNIST indices the split uses.)

- [ ] **Step 2: Run.** `uv run pytest tests/test_mmnist.py -v`. Expected: FAIL. MNIST downloads to `data/` on first run.
- [ ] **Step 3: Implement `mmnist.py`.** Pre-render all frames once in `__init__` (16k × 6 × 64 × 64 float32 ≈ 1.6 GB is too much, so store `uint8` or render lazily per item; *implementer's choice*).
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: stochastic MovingMNIST generator (Q-JEPA F.2)"`

---

### Task 6: CNN encoder + predictor

**Files:**
- Create: `src/q2jepa/encoder/__init__.py`, `src/q2jepa/encoder/cnn.py`, `src/q2jepa/predictor.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `q2jepa.encoder.prepare(x: Tensor, res: int) -> Tensor`: `(B, 3, 64, 64)` → `(B, 3, res, res)` via `F.interpolate(mode="area")`. Identity at 64.
  - `CNNEncoder(res: int, d: int = 64)`: input `(B, 3, res, res)` → `(B, d)`. Conv 3→24 (5×5, s2, p2), 24→48 (3×3, s2, p1), 48→64 (3×3, s2, p1), each followed by SiLU. Then flatten → Linear→96 → SiLU → Linear→d → LayerNorm(d). The flatten width is derived from `res` (64·(res/8)²).
  - `Predictor(d: int = 64, n_bits: int = 3, width: int = 96)`: `forward(h: (B, d), z: (B, K, n)) -> (B, K, d)`. Q-JEPA Eqs. 5–6: `u(z) = σ(W_z2 σ(W_z1(2z−1)))` with widths n→width→width; `c(h) = W_h h` (d→width); `t̂ = LN(h + W_2 σ(W_1[c; u; c⊙u]))` with W_1 3·width→width and W_2 width→d; σ = SiLU.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from q2jepa.encoder import prepare
from q2jepa.encoder.cnn import CNNEncoder
from q2jepa.predictor import Predictor

def test_encoder_shapes_all_res():
    x = torch.rand(5, 3, 64, 64)
    for r in (16, 32, 64):
        assert CNNEncoder(r)(prepare(x, r)).shape == (5, 64)
    assert torch.equal(prepare(x, 64), x)

def test_predictor_shape_and_code_dependence():
    p = Predictor(); h = torch.randn(4, 64)
    z = torch.tensor([[0., 0, 0], [1, 0, 0]]).expand(4, 2, 3)
    out = p(h, z)
    assert out.shape == (4, 2, 64) and not torch.allclose(out[:, 0], out[:, 1])
```

- [ ] **Step 2: Run.** `uv run pytest tests/test_models.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: CNN encoder baseline and shared code predictor"`

---

### Task 7: Classical priors

**Files:**
- Create: `src/q2jepa/prior/__init__.py`, `src/q2jepa/prior/codes.py`, `src/q2jepa/prior/classical.py`
- Test: `tests/test_priors.py`

**Interfaces:**
- Produces:
  - `codes.index_to_bits(idx: LongTensor, n: int) -> Tensor` (float {0,1}, `bit j = (idx >> j) & 1`) and `codes.bits_to_index(z: Tensor) -> LongTensor`.
  - `SoftmaxCategorical(d=64, hidden=48, n_bits=3)`: LN → Linear d→Hp → SiLU → Linear Hp→2ⁿ.
  - `MixturePrior(d=64, hidden=48, n_bits=3, components=2)`: LN → Linear d→Hp → SiLU → Linear Hp→R(n+1) (R component logits, then R·n Bernoulli logits). `log_prob` uses logsumexp over components.
  - Both expose `sample(h: (B, d), k: int) -> Tensor (B, k, n)` (under `torch.no_grad`) and `log_prob(h: (B, d), z: (B, k, n)) -> Tensor (B, k)`.
  - `build_prior(cfg: PriorCfg, d: int) -> nn.Module` dispatches on `cfg.name`.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from q2jepa.prior.codes import index_to_bits, bits_to_index
from q2jepa.prior.classical import SoftmaxCategorical, MixturePrior

def test_bit_order_matches_cudaq():
    assert index_to_bits(torch.tensor([1, 6]), 3).tolist() == [[1, 0, 0], [0, 1, 1]]
    assert bits_to_index(torch.tensor([[1., 0, 0], [0, 1, 1]])).tolist() == [1, 6]

def test_mixture_param_count_matches_paper():
    assert sum(p.numel() for p in MixturePrior().parameters()) == 3640

def test_categorical_forced_code():
    p = SoftmaxCategorical(); torch.nn.init.zeros_(p.net[-1].weight)
    with torch.no_grad(): p.net[-1].bias.copy_(torch.tensor([-50., 50, -50, -50, -50, -50, -50, -50]))
    z = p.sample(torch.randn(2, 64), 16)
    assert (z == torch.tensor([1., 0, 0])).all()

def test_log_probs_normalize():
    h = torch.randn(3, 64); allz = index_to_bits(torch.arange(8), 3).expand(3, 8, 3)
    for prior in (SoftmaxCategorical(), MixturePrior()):
        assert torch.allclose(prior.log_prob(h, allz).exp().sum(-1), torch.ones(3), atol=1e-5)
```

(`net` is an `nn.Sequential` attribute; its last element is the output Linear.)

- [ ] **Step 2: Run.** `uv run pytest tests/test_priors.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: Softmax-Categorical and Mixture-2 classical priors"`

---

### Task 8: Losses

**Files:**
- Create: `src/q2jepa/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Produces (ρ(u, v) = ‖u − v‖₂/√d throughout):
  - `energy_score(pred: (B, K, d), target: (B, d)) -> Tensor` (scalar, batch mean). Q-JEPA Eq. 7.
  - `loo_cost(pred, target) -> Tensor (B, K)`. Eq. 12.
  - `visreg_terms(v: (B, d), n_proj: int = 128) -> tuple[Tensor, Tensor, Tensor]`: (center, scale, shape) exactly as in Q-JEPA App. A.3. Uses population std, `sg(s)` only in the shape normalization, ε = 1e-6, Gaussian quantiles `Φ⁻¹(i/(B+1))` via `torch.special.ndtri`, and fresh random unit directions per call.
  - `visreg(v, cfg: VISRegCfg, it: int) -> Tensor` = `w_center·center + w_scale·scale + min(1, it/shape_ramp)·shape`.
  - `prior_score_loss(log_prob: (B, K), cost: (B, K)) -> Tensor` = `mean(log_prob · cost.detach())`. Eq. 13.

- [ ] **Step 1: Write the failing tests**

```python
import math, torch
from q2jepa.config import VISRegCfg
from q2jepa.losses import energy_score, loo_cost, visreg_terms, visreg, prior_score_loss

def test_energy_score_known_values():
    t = torch.zeros(1, 4)
    assert energy_score(t[:, None].expand(1, 3, 4), t).item() == 0.0
    pred = torch.tensor([[[2., 0, 0, 0], [0, 2, 0, 0]]])
    # fidelity = 1, diversity = 0.5*sqrt(8)/2 = sqrt(2)/2
    assert math.isclose(energy_score(pred, t).item(), 1 - math.sqrt(2) / 2, rel_tol=1e-6)

def test_loo_cost_shape_and_value():
    pred = torch.tensor([[[2., 0, 0, 0], [0, 2, 0, 0]]])
    c = loo_cost(pred, torch.zeros(1, 4))
    assert c.shape == (1, 2) and torch.allclose(c, torch.full((1, 2), 1 - math.sqrt(2)))

def test_visreg_gaussian_vs_collapsed():
    torch.manual_seed(0)
    c, s, sh = visreg_terms(torch.randn(4096, 64))
    assert c < 1e-3 and s < 1e-3 and sh < 1e-2
    _, s0, _ = visreg_terms(torch.ones(64, 64))
    assert math.isclose(s0.item(), 1.0, rel_tol=1e-5)

def test_shape_ramp():
    v = torch.randn(64, 64); cfg = VISRegCfg()
    torch.manual_seed(1); a = visreg(v, cfg, 0)
    c, s, _ = visreg_terms(v)
    assert torch.isclose(a, 1.0 * c + 10.0 * s, atol=1e-6)

def test_prior_score_loss_detaches_cost():
    lp = torch.zeros(2, 3, requires_grad=True); cost = torch.ones(2, 3, requires_grad=True)
    prior_score_loss(lp, cost).backward()
    assert cost.grad is None and torch.allclose(lp.grad, torch.full((2, 3), 1 / 6))
```

- [ ] **Step 2: Run.** `uv run pytest tests/test_losses.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: energy score, LOO cost, VISReg and prior score-function loss"`

---

### Task 9: Translation readout + metrics

**Files:**
- Create: `src/q2jepa/readout/__init__.py`, `src/q2jepa/readout/mmnist.py`, `src/q2jepa/metrics.py`
- Test: `tests/test_readout_metrics.py`

**Interfaces:**
- Consumes: `translate`, `true_trajectories`, `DIRECTION_PROBS` (Task 5).
- Produces:
  - `TranslationReadout(d=64, hidden=96, frames=3, bound=18.0)`: `forward(h, t) -> disp (B, 3, 2)`. Attribute `net = nn.Sequential(Linear 2d→96, SiLU, Linear 96→6)`, with the last Linear zero-initialized, then `bound·tanh(a/bound)`.
  - `render(last_frame: (B, 64, 64), disp: (B, 3, 2)) -> (B, 3, 64, 64)` via `translate`.
  - `readout_loss(pred, target) -> Tensor` = `mean((1+4y)|ŷ−y|) + 0.5·(1 − soft Dice)`, Dice smoothing 1e-6, averaged over frames.
  - `mmnist_metrics(disp: np.ndarray (S, 3, 2)) -> dict[str, float]` with keys `coverage, validity, tv, min_ade, es` (Q-JEPA F.4). Assign each sample to the direction with the nearest final (f = 3) endpoint. It is valid if that error is < 4 px. TV is taken over 8 directions plus invalid (target p_invalid = 0). minADE = Σ_k p_k · min_s mean_f ‖τ̂_s[f] − τ_k[f]‖. ES = Σ_k p_k · mean_s ‖vec τ̂_s − vec τ_k‖ − ½ · mean_{s≠s'} ‖vec τ̂_s − vec τ̂_s'‖.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, torch
from q2jepa.data.mmnist import true_trajectories
from q2jepa.readout.mmnist import TranslationReadout, render
from q2jepa.metrics import mmnist_metrics

def test_zero_init_identity_and_bound():
    r = TranslationReadout(); d = r(torch.randn(2, 64), torch.randn(2, 64))
    assert torch.equal(d, torch.zeros(2, 3, 2))
    last = torch.rand(2, 64, 64)
    assert torch.allclose(render(last, d), last[:, None].expand(2, 3, 64, 64), atol=1e-6)
    with torch.no_grad(): r.net[-1].bias.fill_(1e6)
    assert r(torch.randn(1, 64), torch.randn(1, 64)).abs().max() <= 18.0

def test_metrics_perfect_distribution():
    t = true_trajectories()
    samples = np.concatenate([np.repeat(t[k:k+1], 3 if k % 2 == 0 else 2, 0) for k in range(8)])
    m = mmnist_metrics(samples)
    assert m["validity"] == 1.0 and m["coverage"] == 1.0 and abs(m["tv"]) < 1e-9 and abs(m["min_ade"]) < 1e-9

def test_metrics_single_mode_and_invalid():
    t = true_trajectories()
    m = mmnist_metrics(np.repeat(t[0:1], 128, 0))
    assert m["coverage"] == 1 / 8 and abs(m["tv"] - 0.85) < 1e-9
    far = mmnist_metrics(np.full((128, 3, 2), 50.0))
    assert far["validity"] == 0.0 and far["coverage"] == 0.0 and abs(far["tv"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run.** `uv run pytest tests/test_readout_metrics.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: frozen translation readout and Q-JEPA MovingMNIST metrics"`

---

### Task 10: Training loop, smoke test, classical acceptance run

**Files:**
- Modify: `src/q2jepa/train.py`, `src/q2jepa/encoder/__init__.py` (add `build_encoder(cfg: EncoderCfg) -> nn.Module`, only `"cnn"` for now)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `stage(it: int, warmup_repr: int, warmup_pred: int) -> Literal["repr", "pred", "full"]`, 1-indexed. `it ≤ warmup_repr` → "repr", `≤ warmup_pred` → "pred", otherwise "full".
  - `run(cfg: Config, out_dir: Path) -> dict[str, float]`, which does the whole pipeline and writes `out_dir/metrics.json` and `out_dir/checkpoint.pt`. `main` calls `run(cfg, HydraConfig.get().runtime.output_dir)`.
- Behaviour (Q-JEPA §3, App. A.3 "Training sequence"):
  1. Seed `torch`, `numpy` and `random` from `cfg.seed`. Call `set_target(cfg.quantum.target)`. Build the encoder, prior and predictor. Use AdamW with param groups {encoder: lr_enc, predictor: lr_pred, prior: lr_prior}, all with weight_decay.
  2. Each iteration: get a batch of (obs, fut). Compute `h = enc(prepare(obs))` and `t = enc(prepare(fut))`.
     - "repr": loss = λ_reg·(R(h) + R(t))/2.
     - "pred": additionally sample `z ~ prior.sample(h.detach(), K)`, compute `t̂ = pred(h, z)`, and add `energy_score(t̂, t.detach())`.
     - "full": additionally add `prior_score_loss(prior.log_prob(h.detach(), z), loo_cost(t̂, t).detach())`.
     - Clip each group's grad norm at `clip`. Set the encoder LR to `lr_enc_later` from iteration `warmup_repr + 1`.
  3. Freeze the JEPA parameters. Train `TranslationReadout` on `[h; t]` from real targets for `readout.iters`, with loss `readout_loss(render(obs[:, 2], disp), fut)`.
  4. Evaluate on the test split's `unique_obs()[:eval.n_obs]`: `k_eval` prior samples → predictor → readout `(h, t̂)` → `mmnist_metrics` per observation, averaged.
  5. Log to wandb (`mode=cfg.wandb.mode`). Write the metrics JSON (sorted keys).
- **Checkpoint/resume (Colab).** New `TrainerCfg` fields: `ckpt_every: int = 500` and `stop_after: int = 0`. `stop_after` ends this session after that many world-model iterations; 0 means run to the end. Colab users chunk long runs with it.
  - Every `ckpt_every` iterations, and when `stop_after` triggers, atomically write `out_dir/ckpt.pt`: write to `ckpt.pt.tmp`, then `os.replace`. It holds encoder/predictor/prior/optimizer state dicts and `it`.
  - Write `out_dir/readout.pt` once readout training finishes.
  - On entry, `run` resumes from whatever exists: skip finished world-model iterations, and skip readout training if `readout.pt` exists.
  - To make resume exact without saving RNG/iterator state, **every world-model iteration reseeds**: `torch.manual_seed(cfg.seed * 1_000_003 + it)`. Batch indices come from `torch.randint` under that seed, not from a DataLoader iterator. Readout iterations do the same with offset `+ 10_000_000`.
  - When `stop_after` triggers, `run` returns `{}` and writes no metrics.

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path
from hydra import compose, initialize_config_dir
import q2jepa.config  # noqa: F401
from q2jepa.train import stage, run

CONF = str(Path(__file__).resolve().parents[1] / "conf")
TINY = ["trainer.iters=6", "trainer.warmup_repr=2", "trainer.warmup_pred=4", "trainer.batch=8", "trainer.k_train=4",
        "readout.iters=2", "readout.batch=8", "eval.n_obs=2", "eval.k_eval=8",
        "dataset.n_train=2", "dataset.n_test=2", "wandb.mode=disabled"]

def test_stage_boundaries():
    assert [stage(i, 200, 300) for i in (1, 200, 201, 300, 301)] == ["repr", "repr", "pred", "pred", "full"]

def test_smoke_and_reproducible(tmp_path):
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose("config", overrides=TINY)
    m1 = run(cfg, tmp_path / "a"); m2 = run(cfg, tmp_path / "b")
    assert set(m1) >= {"coverage", "validity", "tv", "min_ade", "es"}
    assert (tmp_path / "a" / "metrics.json").read_text() == (tmp_path / "b" / "metrics.json").read_text()
    assert json.loads((tmp_path / "a" / "metrics.json").read_text()) == m1

def test_interrupted_run_resumes_to_same_metrics(tmp_path):
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        full = compose("config", overrides=TINY + ["trainer.ckpt_every=2"])
        part = compose("config", overrides=TINY + ["trainer.ckpt_every=2", "trainer.stop_after=3"])
    ref = run(full, tmp_path / "ref")
    assert run(part, tmp_path / "res") == {} and (tmp_path / "res" / "ckpt.pt").exists()
    assert not (tmp_path / "res" / "metrics.json").exists()
    assert run(full, tmp_path / "res") == ref
```

- [ ] **Step 2: Run.** `uv run pytest tests/test_train.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `stage`, `run` and `main`.**
- [ ] **Step 4: Run.** `uv run ruff check . && uv run pytest -v` (full suite). Expected: no lint errors, all PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: 3-stage JEPA training, readout and evaluation pipeline"`
- [ ] **Step 6: Acceptance run, paper reproduction** (64×64 CNN, Softmax-Categorical, seed 0):

Run: `uv run python -m q2jepa.train encoder.res=64 seed=0`
Expected `metrics.json`: `validity ≥ 0.85`, `coverage ≥ 0.80`, `tv ≤ 0.25`. The paper reports 95.8 / 93.6 / 0.145. If it misses, stop and use superpowers:systematic-debugging before continuing. Do not tune toward the test split; use `dev`.

- [ ] **Step 7: Record the downsampled baseline.** `uv run python -m q2jepa.train encoder.res=16 seed=0` and `prior=mixture encoder.res=64 seed=0`. Write all three `metrics.json` results into `docs/results/2026-10-kickstart-classical.md` (a table with one row per run and the git SHA). There are no thresholds for these two runs.
- [ ] **Step 8: Commit.** `git add docs/results && git commit -m "docs: classical MovingMNIST baseline results"`

---

### Task 11: Attention benchmark + Colab GPU notebook

**Files:**
- Create: `scripts/__init__.py`, `scripts/bench_attention.py`, `scripts/colab_setup.py`, `notebooks/colab_gpu.ipynb`
- Test: `tests/test_bench.py`, `tests/test_colab_setup.py`

**Interfaces:**
- Consumes: `factored_acm`, `full_swap_test_probs`, `image_to_patches`, `set_target`.
- Produces: `run_bench(target: str, res: int, n_images: int, seed: int = 0) -> list[dict]`. There is one dict per mode ∈ {"factored", "full"} with keys `target, res, mode, qubits, ms_per_image`. `qubits` is `m` for factored and `2·log2(res²)+1` for full (17 at res 16, 21 at res 32). The CLI is `--target`, `--res` (repeatable), `--images`, and it prints a markdown table.

- [ ] **Step 1: Write the failing test**

```python
from scripts.bench_attention import run_bench

def test_bench_rows():
    rows = run_bench("qpp-cpu", 16, 1)
    assert {r["mode"] for r in rows} == {"factored", "full"}
    assert {r["qubits"] for r in rows} == {4, 17} and all(r["ms_per_image"] > 0 for r in rows)
```

(Add an empty `scripts/__init__.py` so the import resolves.)

- [ ] **Step 2: Run.** `uv run pytest tests/test_bench.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Use random images from a fixed seed and parameters from `n_ansatz_params(4, 2)`. Time with `time.perf_counter` after one warm-up call.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: swap-test attention CPU/GPU benchmark script"`
- [ ] **Step 6: Colab setup helper (TDD).** `scripts/colab_setup.py` provides `cuda_major(nvidia_smi_output: str) -> int`, which parses `CUDA Version: X.Y` from the `nvidia-smi` header, and `cudaq_wheel(cuda_major: int) -> str`, which returns `"cuda-quantum-cu13==0.16.0"` for ≥ 13, `"cuda-quantum-cu12==0.16.0"` for 12, and raises `RuntimeError` below 12. `main()` runs `nvidia-smi`. If the needed wheel isn't the installed one, it pip-uninstalls `cuda-quantum-cu13` and installs the cu12 wheel. Tests:

```python
import pytest
from scripts.colab_setup import cuda_major, cudaq_wheel

def test_parse_and_pick():
    hdr = "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4     |"
    assert cuda_major(hdr) == 12 and cudaq_wheel(12) == "cuda-quantum-cu12==0.16.0"
    assert cudaq_wheel(13) == "cuda-quantum-cu13==0.16.0"
    with pytest.raises(RuntimeError):
        cudaq_wheel(11)
```

Run it, watch it fail, implement, watch it pass, then commit `feat: colab CUDA-Q wheel selection helper`.

- [ ] **Step 7: Write `notebooks/colab_gpu.ipynb`.** It is a thin driver, with each cell being a few lines and no model code:
  1. `!nvidia-smi`, and mount Drive (`/content/drive`).
  2. Set `REPO_URL` (the team fills it in once the repo is on GitHub), `!git clone`, `%cd`, `!pip install -q -e .`, `!python scripts/colab_setup.py`.
  3. `!Q2JEPA_CUDAQ_TARGET=nvidia python -m pytest tests/test_born.py tests/test_attention_equivalence.py -q`
  4. `!python scripts/bench_attention.py --target qpp-cpu --res 16 --res 32 --images 16` and the same with `--target nvidia`.
  5. Compose the config with `hydra.compose(overrides=["quantum=gpu", "trainer.device=cuda", "encoder.res=64", "seed=0"])` and call `run(cfg, Path("/content/drive/MyDrive/q2jepa/runs/classical-res64-s0"))`. Re-running the cell after a disconnect resumes from the Drive checkpoint.

  Verify locally that the notebook is valid with `uv run python -c "import json; json.load(open('notebooks/colab_gpu.ipynb'))"`. Commit with `feat: colab GPU driver notebook`.

- [ ] **Step 8: Colab verification (manual, the team runs it).** Expected: the quantum tests PASS on `nvidia`. Both benchmark tables print, with full mode at 17 and 21 qubits. The training run's validity is within ±0.05 of the Mac acceptance run. Paste the tables and the GPU model (`nvidia-smi --query-gpu=name --format=csv`) into `docs/results/2026-10-bench-attention.md` and commit `docs: first CUDA-Q CPU vs GPU attention benchmark`.

---

**Deferred to the next plans (not gaps):** HQViT encoder module and its gradient path, the QCBM prior module with parameter-shift updates (`quantum/bridge.py`), the Dice metric, the `[ibm]` extra and Qiskit mirror, and Highway.
