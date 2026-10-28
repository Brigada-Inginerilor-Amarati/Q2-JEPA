# Q²-JEPA — Tech Stack & Repo Kickstart Design

Date: 2026-10-28 · Status: draft for team review

## 1. Intent

We are building Q²-JEPA for a one-month AI × HPC × Quantum hackathon. It is a visual world model that predicts *several* plausible futures from a short video clip. It combines two quantum components:

1. **HQViT encoder** (Zhang et al., arXiv:2504.02730). Whole-image amplitude encoding plus swap-test quantum self-attention. It replaces the small CNN encoder used in the Q-JEPA paper.
2. **Conditional Born-machine prior** (Liu et al., Q-JEPA, arXiv:2610.27921). An entangled QCBM, a quantum circuit that we sample from, produces binary latent action codes. A shared predictor turns each code into a different future representation.

All quantum execution is **simulated with NVIDIA CUDA-Q on GPUs**. That is the HPC story. The full architecture is specified in `docs/Q2_JEPA_Architecture_Slide.pdf`. The pitch deck is `docs/HQViT x Q-JEPA.pdf`. Both source papers are in `docs/`.

This document covers **the tech stack and how the repo starts out**. It is not a model-training plan.

## 2. Goals, non-goals, anti-goals

**Goals (team-decided)**
- Benchmark results on **MovingMNIST**, through the visual HQViT encoder.
- **Highway-Env is deferred.** It will use rendered frames when it comes back, but the task definition (how to get several ground-truth futures per scene) is undecided. Nothing in this phase builds Highway code; the layout only needs to leave room for a second dataset.
- Everything in **Python**, with quantum circuits **simulated via CUDA-Q**.
- **IBM quantum hardware: inference only.** Sample the *trained* Born prior on a real QPU and compare the metrics against GPU-simulated sampling.

**Goals (my calls)**
- A classical-vs-quantum comparison grid, so results mean something: encoder {CNN, HQViT} × prior {classical, QCBM}.
- Every result-table row can be reproduced from a saved config and seed.

**Non-goals**
- Training anything on IBM hardware, or running the HQViT encoder on hardware. Shot counts and queue latency make it infeasible: about 1.28B simulated shots per training run in the paper.
- D3IL. It's in the deck but not in "done". Pick it up only if time is left.
- Splitting the repo or the code ownership across the five team members.

**Anti-goals (my calls: things that would count as failure even if the code runs)**
- Quantum results with no classical baseline trained under identical settings.
- A "quantum" path that quietly bypasses CUDA-Q, such as a hand-written torch statevector that ends up in reported runs.
- Results that can't be regenerated, for example a manual notebook tweak that isn't in config.

## 3. Constraints

- **Timeline:** one month.
- **Hardware:** dedicated NVIDIA GPU access is limited, so GPU work runs on **Google Colab** (Python 3.12, sessions disconnect, GPU type not guaranteed). Training must therefore **checkpoint to Google Drive and resume**. Team laptops are Apple-silicon Macs. IBM QPUs are available for inference.
- **CUDA-Q platform support (verified 2026-10-28):** `cudaq` 0.16.0 installs and runs on this Mac (Apple silicon) under Python 3.14 with the `qpp-cpu` target, which is **CPU simulation only**. GPU targets (`nvidia`, `nvidia-mqpu`) are Linux-only, and Linux wheels exist for cp311–cp314. torch 2.15, hydra-core 1.3, wandb 0.31, qiskit 2.6 and qiskit-ibm-runtime 0.51 were also verified on Python 3.14.
- **CUDA-Q API churn:** 0.16 warns that the `sample` and `observe` primitives "will change in a future release". Pin `cudaq` to `~=0.16.0`.
- **IBM access:** the free Open Plan allows 10 QPU minutes per 28 days, and paid plans cost $48–96 per minute. Hackathon-provided access may differ. Hardware runs must be batched and small.

## 4. Tech stack

| Concern | Choice | Who decided |
|---|---|---|
| Language | Python **3.12** across local dev and Colab (`.python-version` = 3.12, `requires-python = ">=3.12"`) | team |
| Env & deps | `uv` + `pyproject.toml` + `uv.lock` locally. On Colab, `pip install -e .` into the preinstalled environment. Dependency floors stay loose (e.g. torch) so Colab's preinstalled packages satisfy them. | team (loose torch) / me |
| Quantum SDK pin | CUDA-Q 0.16.0 via the concrete wheel `cuda-quantum-cu13==0.16.0`, which has wheels for macOS arm64 and Linux. The `cudaq` package on PyPI is a source package whose dependency is chosen when it is built on each machine, which breaks uv's cross-platform lock, and `cudaq` 0.16.0.post1 has no macOS wheel. On a GPU with a CUDA-12 driver, swap in `cuda-quantum-cu12==0.16.0` (same API). | team (0.16.x) / me (wheel) |
| GPU runtime | **Google Colab** GPU runtimes, driven by a thin notebook (`notebooks/colab_gpu.ipynb`) that installs this package and calls into it. No code lives in the notebook. | team |
| Classical DL | PyTorch | me |
| Quantum simulation | `cudaq` Python kernels. Target set by config: `qpp-cpu` on Mac, `nvidia` on GPU boxes. | team (CUDA-Q) / me (targets) |
| Configs | **Hydra** with dataclass schemas registered in `ConfigStore` | discussed, team leaning yes |
| Experiment tracking | Weights & Biases, falling back to offline mode | me |
| Tests | `pytest`, used only for correctness-critical pieces | me |
| Lint/format | `ruff` | me |
| Benchmark | Stochastic MovingMNIST generated from torchvision MNIST (no `highway-env` dependency yet) | team (benchmark) / me (source) |
| IBM inference | `qiskit` + `qiskit-ibm-runtime` (`SamplerV2`), in an **optional** `[ibm]` dependency group | team (inference only) / me (packaging) |
| GPU runtime fallback | NVIDIA CUDA-Q container (`nvcr.io/nvidia/quantum/cuda-quantum`) if the pip install fails on the cluster | me |

Why Hydra: the experiment grid is a cross-product. Encoder {CNN, HQViT} × prior {Softmax-Categorical, Mixture, product-QCBM, entangled-QCBM, frozen} × gradient {ideal, param-shift, param-shift+noise} × resolution × seeds, with a dataset axis to come.

**Core results set (my proposal; GPU budget still unknown).** Encoder {CNN, HQViT} × prior {best classical, entangled-QCBM with noisy param-shift} × 3 seeds on MovingMNIST = 12 runs. Everything else in the grid is stretch. Config groups plus `-m` multirun cover this without hand-written sweep scripts, and each run's output folder stores its resolved config. Add the `submitit` launcher only if the GPUs sit behind SLURM.

## 5. Repo layout

One shared package, organized by part of the model:

```
Q2-JEPA/
├── pyproject.toml            # uv-managed; extras: [ibm], [dev]
├── conf/                     # Hydra
│   ├── config.yaml           # defaults list
│   ├── encoder/   {cnn,hqvit}.yaml
│   ├── prior/     {classical,product_qcbm,entangled_qcbm,frozen}.yaml
│   ├── dataset/   mmnist.yaml
│   ├── quantum/   {cpu,gpu}.yaml        # cudaq target, shots, noise, grad mode
│   └── trainer/   default.yaml          # Table 5 settings from the Q-JEPA paper
├── src/q2jepa/
│   ├── quantum/              # ALL cudaq code lives here
│   │   ├── attention.py      # swap-test attention kernels (full 17q + factored 4q)
│   │   ├── born.py           # conditional entangled-QCBM sampling kernel
│   │   └── bridge.py         # torch <-> cudaq glue (batching, param-shift)
│   ├── encoder/              # CNN baseline, HQViT (QTB blocks, FFN, pooling, temporal fusion)
│   ├── prior/                # angle conditioner g_eta, classical priors, QCBM wrapper
│   ├── predictor.py          # P_psi(h, z)
│   ├── losses.py             # energy score (LOO), VISReg
│   ├── readout/              # frozen-JEPA decoders per dataset
│   ├── data/                 # mmnist.py (stochastic variant generator + loader)
│   ├── metrics.py            # coverage, validity, TV, minADE, ES, Dice
│   └── train.py              # Hydra entrypoint; 3-stage schedule
├── scripts/
│   ├── bench_attention.py    # HPC benchmark: full vs factored, CPU vs GPU
│   └── eval_hardware.py      # IBM inference (imports qiskit; nothing else does)
├── tests/
│   ├── test_attention_equivalence.py  # factored 4q == full 17q swap test
│   ├── test_born_equivalence.py       # cudaq vs qiskit Born probs (bit order)
│   └── test_losses.py                 # VISReg / energy score sanity
└── docs/                     # papers, slides, specs
```

Rules (my calls):
- `cudaq` is imported **only** under `src/q2jepa/quantum/`.
- `qiskit` is imported **only** in `scripts/eval_hardware.py` and its test.
- Adding a benchmark means adding a `data/` module, a `readout/` module, a `conf/dataset/` yaml and metrics. It never means a new training script.

## 6. Key technical decisions

**Two execution modes for HQViT attention (from the slide; which mode does what is my call).**
- *Factored 4-qubit mode*: exact. It works because the ansatz is `I_idx ⊗ W(θ)` and the swap test acts only on patch qubits, so the joint distribution splits into per-patch circuits. This is the **training path**.
- *Full 17-qubit swap-test mode* (21q at 32×32): shot-based and faithful to the HQViT paper. Used for the equivalence test and the HPC benchmark (`bench_attention.py`: wall-clock and shots, CPU vs GPU, full vs factored). A short training run in this mode is optional.

**Born prior gradients.** These follow the Q-JEPA paper, Section 3.4 and Appendix A. The prior update is computed **separately** from backprop:
- In parameter-shift mode, run shifted-circuit sampling through CUDA-Q with subsampling (b, r) and the 0.5% readout-flip channel.
- Ideal-gradient mode is the reference.

How the HQViT encoder parameters (θQ, θK, θV) get gradients through CUDA-Q is **left to the builder**. Options include a parameter-shift `torch.autograd.Function` or statevector-based gradients. Constraint: it must run through `cudaq`, and it must fit the step budget on the GPUs.

**Data.**
- *MovingMNIST*: the Q-JEPA paper's **stochastic single-digit variant** (Appendix F.2): 3 observation and 3 future frames, 8 future directions, metrics coverage/validity/TV/minADE/ES/Dice. This is what makes the Born prior's multimodality measurable, and it lets us compare with the paper's Table 2. *(My call. The classic deterministic two-digit MovingMNIST would not exercise the prior.)*
- *Resolution*: a config value. 16×16 (8q registers, 17q full swap test) vs 32×32 (10q, 21q). Test both early. 64×64 (12q registers, 25q full) is out of scope. *(My call.)* Only the **encoder input** is downsampled. Data generation, readout and metrics stay at native resolution (64×64 for MovingMNIST, so the paper's 4-pixel threshold and Table 2 comparison still hold). **Every encoder in the grid, CNN included, sees the same downsampled input.** The paper's 64×64 CNN runs once as a sanity check that our pipeline reproduces its numbers, and is not a grid row.

**Classical priors.** *(My call.)* Softmax-Categorical (the paper's strongest on frequency accuracy) and the parameter-matched Mixture (R=2). Other priors from the paper (AR-GRU, independent bits) are not built.

**HQViT-full is not a grid row.** *(My call; this resolves the conflict with "optional" above.)* Full 17q mode appears only in the equivalence test and `bench_attention.py`.

**Hyperparameters.** Start from Q-JEPA Table 5: d = 64, predictor width 96, Hp = 48, MovingMNIST 3 bits / L = 2, 4,000 iterations, batch 64, K = 32 train / 128 eval, warm-up 200/300, AdamW, grad-clip 5, 256 shots per shifted circuit.

**IBM inference.** *(My call.)* MovingMNIST's 3-qubit prior. The team still needs to confirm which backend and how many minutes the hackathon provides, but that only matters in week 4. The Born circuit is hand-mirrored in Qiskit (`ParameterVector`). `test_born_equivalence.py` checks that both produce the same probabilities **including bit order**, because Qiskit is little-endian. `eval_hardware.py` loads a checkpoint, computes angles for N test observations, submits them as one `SamplerV2` job, and writes the same metrics as simulated evaluation.

## 7. Kickstart sequence (first days)

1. `git init`, `uv init`, add dependencies, add `.gitignore` (data, outputs, wandb). Commit.
2. Hydra skeleton: `train.py` prints the resolved config. Config groups exist with the Table 5 values.
3. `quantum/born.py` + `quantum/attention.py` as minimal kernels. Equivalence tests pass on Mac CPU.
4. Classical end-to-end on stochastic MovingMNIST: CNN encoder + classical prior + predictor + losses + readout + metrics. Sanity-check it against the paper's classical numbers.
5. Verify on Colab via `notebooks/colab_gpu.ipynb`: run the quantum tests on the `nvidia` target, produce the first CPU-vs-GPU numbers from `bench_attention.py`, and do the same training run with `quantum=gpu`, checkpointing to Drive.

After that come quantum prior → HQViT encoder → IBM inference, all from the implementation plan.

## 8. Still open (team)

- **GPU budget**: GPU count, hours, and whether there is SLURM. This sizes the run grid and the encoder-gradient approach. It isn't needed for steps 1–4 of the kickstart.
- **Highway task definition**: deferred (see section 2).
- **IBM access details**: backend and minutes, needed by week 4.

## 9. Left to the builder

- The encoder-gradient strategy through CUDA-Q (see section 6).
- How the 3-frame clip enters HQViT: amplitude-encode each frame and fuse after pooling ("Temporal Fusion" on the slide), or encode frames jointly.
- W&B project and entity naming, and how the CI-less test workflow runs (local `uv run pytest`).
