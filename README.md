# Q²-JEPA

**Q²-JEPA** is a dual-quantum visual world model that predicts multimodal plausible futures from short video clips, simulated on GPUs with **NVIDIA CUDA-Q**:

1. **HQViT Encoder** ([Zhang et al., arXiv:2504.02730](docs/papers/hqvit.pdf)) — whole-image amplitude encoding and swap-test quantum self-attention (factored 4-qubit training mode and full 17/21-qubit swap-test benchmark mode).
2. **Conditional Born-Machine Prior** ([Liu et al., Q-JEPA, arXiv:2610.27921](docs/papers/qjepa.pdf)) — an entangled Quantum Circuit Born Machine (QCBM) that samples binary latent action codes $z$ so a shared predictor produces diverse future representations.

## Setup

Requires Python 3.12 (pinned in `.python-version`) and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
lefthook install
```

## Usage

### Train & Evaluate (3-Stage JEPA + Readout)

Runs are configured with Hydra (`conf/config.yaml`) and support deterministic checkpointing and resume (`checkpoint_dir`):

```bash
# Default classical baseline (CNN encoder @ 16x16, Softmax-Categorical prior, CPU target)
uv run python -m q2jepa.train

# Full 64x64 sanity baseline or Mixture-2 prior
uv run python -m q2jepa.train encoder.res=64 seed=0
uv run python -m q2jepa.train encoder.res=64 prior=mixture seed=0

# GPU simulation with checkpointing
uv run python -m q2jepa.train quantum=gpu trainer.device=cuda checkpoint_dir=outputs/ckpt
```

### Swap-Test Attention Benchmark

Compare the exact factored $m$-qubit kernels against the full $(2n+1)$-qubit swap-test circuit across CPU and GPU targets:

```bash
uv run python scripts/bench_attention.py --target qpp-cpu --res 16 --res 32 --images 16
```

### Google Colab GPU Workflow

Use [`notebooks/colab_gpu.ipynb`](notebooks/colab_gpu.ipynb) (backed by [`scripts/colab_setup.py`](scripts/colab_setup.py)) in the Colab web UI or from the terminal via [`google-colab-cli`](https://pypi.org/project/google-colab-cli/) (`colab`) to install the matching CUDA 12/13 `cuda-quantum` wheel on Colab GPU runtimes, run tests and benchmarks on the `nvidia` target, and checkpoint training runs to Google Drive.

#### Running via the `colab` CLI

```bash
# 1. Install the CLI and verify authentication
uv tool install google-colab-cli
colab whoami

# 2. Provision a named GPU session (T4, L4, or A100)
colab new -s q2jepa --gpu T4

# 3a. Mount Google Drive interactively (persists checkpoints to Drive)
colab drivemount -s q2jepa

# 3b. Or, for headless/agent runs without interactive Drive OAuth, create a local fallback directory
echo 'import os; os.makedirs("/content/drive/MyDrive", exist_ok=True)' | colab exec -s q2jepa

# 4. Execute the full notebook (writes cell outputs to notebooks/colab_gpu_output.ipynb)
colab exec -s q2jepa --timeout 600 -f notebooks/colab_gpu.ipynb

# 5. Always release the VM when finished
colab stop -s q2jepa
```

### Development & Hooks

Git `pre-commit` hooks are managed by [Lefthook](https://github.com/evilmartians/lefthook) (`lefthook.yml`) and automatically run `ruff check --fix`, `ruff format`, and `pytest`:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
lefthook run pre-commit
```

## Repository Structure

```text
Q2-JEPA/
├── conf/                     # Hydra configs (dataset, encoder, prior, quantum, trainer)
├── src/q2jepa/
│   ├── config.py             # Dataclass config schemas registered in Hydra ConfigStore
│   ├── quantum/              # CUDA-Q kernels: swap-test attention & conditional Born machine
│   ├── encoder/              # Visual encoders (CNN baseline, HQViT)
│   ├── prior/                # Latent action priors (Softmax-Categorical, Mixture, QCBM)
│   ├── predictor.py          # Latent predictor P_psi(h, z)
│   ├── losses.py             # Leave-one-out energy score, prior score loss, VISReg
│   ├── data/                 # Stochastic single-digit MovingMNIST generator
│   ├── readout/              # Frozen-JEPA translation readout & renderer
│   ├── metrics.py            # Coverage, validity, TV, minADE, Energy Score
│   └── train.py              # 3-stage training, readout, and evaluation entrypoint
├── scripts/                  # Attention HPC benchmark & Colab wheel setup helper
├── notebooks/                # Thin Colab GPU driver notebook
├── tests/                    # Equivalence, prior, loss, readout, and training tests
└── docs/                     # Architecture slides, papers, specs, and baseline results
```

## Documentation & Results

- **Architecture & Pitch**: [`docs/papers/Q2_JEPA_Architecture_Slide.pdf`](docs/Q2_JEPA_Architecture_Slide.pdf) · [`docs/papers/HQViT x Q-JEPA.pdf`](docs/HQViT%20x%20Q-JEPA.pdf)
- **Design Spec**: [`docs/superpowers/specs/2026-10-28-q2jepa-stack-and-repo-design.md`](docs/superpowers/specs/2026-10-28-q2jepa-stack-and-repo-design.md)
- **Classical MovingMNIST Baselines**: [`docs/results/2026-10-kickstart-classical.md`](docs/results/2026-10-kickstart-classical.md)
