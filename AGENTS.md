# AGENTS.md

Guidance for coding agents working in this repo. This file is the single source of truth for agents;
`CLAUDE.md` only imports it. The README is the human quickstart — where they disagree, this file wins.

## 1. What this is

Q²-JEPA: a visual world model that predicts several plausible futures from a short clip, with two quantum
parts simulated on GPUs via NVIDIA CUDA-Q:

- **HQViT encoder** — amplitude encoding + swap-test quantum self-attention ([docs/papers/hqvit.pdf](docs/papers/hqvit.pdf)).
- **Conditional Born-machine prior (QCBM)** — samples binary latent action codes `z` ([docs/papers/qjepa.pdf](docs/papers/qjepa.pdf)).

Read before non-trivial work:

| Doc | What it decides |
|---|---|
| [docs/superpowers/specs/2026-10-28-q2jepa-stack-and-repo-design.md](docs/superpowers/specs/2026-10-28-q2jepa-stack-and-repo-design.md) | Stack, layout, rules, experiment grid, open questions |
| [docs/superpowers/plans/2026-10-28-q2jepa-kickstart.md](docs/superpowers/plans/2026-10-28-q2jepa-kickstart.md) | Kickstart tasks 1–11 (done) and their acceptance criteria |
| [docs/papers/Q2_JEPA_Architecture_Slide.pdf](docs/papers/Q2_JEPA_Architecture_Slide.pdf) | Full target architecture |
| [docs/results/](docs/results/) | Reported numbers; every row reproducible from overrides + seed + SHA |

## 2. Current state vs. target

The README describes the target. What actually exists today:

| Part | Status |
|---|---|
| Stochastic MovingMNIST (`data/mmnist.py`) | done |
| CNN encoder (`encoder/cnn.py`, `encoder=cnn`) | done |
| Classical priors: Softmax-Categorical, Mixture (`prior/classical.py`) | done |
| Predictor, losses, readout, metrics, 3-stage `train.py` with resume | done |
| Swap-test attention kernels, factored 4q + full 2n+1 q (`quantum/attention.py`) | kernels only, not wired into an encoder |
| Born-machine kernel (`quantum/born.py`) | kernel only, not wired into a prior |
| HQViT encoder (`encoder/hqvit.py`, `conf/encoder/hqvit.yaml`) | **not built** |
| QCBM prior + param-shift gradients (`quantum/bridge.py`, `conf/prior/*qcbm*.yaml`) | **not built** |
| IBM inference (`scripts/eval_hardware.py`, `[ibm]` extra) | **not built** |

Do not pass `encoder=hqvit` or `prior=*qcbm*` until the matching config group exists. Next build order
(spec §7): quantum prior → HQViT encoder → IBM inference. Plan new work with a spec/plan under
`docs/superpowers/` like the kickstart.

## 3. Hard rules

From the spec; breaking any of these makes results invalid even if the code runs.

- `cudaq` is imported **only** under `src/q2jepa/quantum/`. Everything else calls into it.
- `qiskit` is imported **only** in `scripts/eval_hardware.py` and its test.
  (Both enforced by `tests/test_import_boundaries.py`.)
- No "quantum" path that bypasses CUDA-Q (e.g. a hand-written torch statevector) in any reported run.
- No quantum result without a classical baseline trained under identical settings (same `encoder.res`, seed, trainer).
- Every encoder in the grid, CNN included, sees the same downsampled input (`encoder.res`); data,
  readout and metrics stay at native 64×64.
- Every reported number must be regenerable from config overrides + seed + commit SHA. No manual
  notebook tweaks: change config or code, not the notebook.
- A new benchmark = a `data/` module + a `readout/` module + `conf/dataset/*.yaml` + metrics. Never a
  new training script.
- No hardware training; IBM QPUs are inference-only on the trained Born prior.
- The notebook stays thin: it installs the package and calls into it. No logic lives there.

## 4. Local dev loop (Mac, CPU)

Python 3.12 (`.python-version`), `uv`. CUDA-Q runs on macOS arm64 with the `qpp-cpu` target only.

```bash
uv sync
lefthook install                      # pre-commit: ruff check --fix, ruff format, pytest
uv run pytest                         # all tests, qpp-cpu target
uv run ruff check . && uv run ruff format .
```

- GitHub Actions (`.github/workflows/ci.yml`) reruns ruff + pytest (`qpp-cpu`) on every push and PR.
- Tests pick the CUDA-Q target from `Q2JEPA_CUDAQ_TARGET` (default `qpp-cpu`, see `tests/conftest.py`).
- `cudaq` kernel bodies use DSL gate names (`rx`, `h`, ...); `F821` is ignored under `quantum/` on purpose.
- Write the failing test first for correctness-critical pieces (kernels, equivalences, losses, metrics).

### Training

Hydra config in `conf/`, typed schema in `src/q2jepa/config.py` (Q-JEPA Table 5 defaults). Unknown keys
are rejected — check the schema before inventing an override.

```bash
uv run python -m q2jepa.train                                   # CNN @16x16, Softmax-Categorical, qpp-cpu
uv run python -m q2jepa.train encoder.res=64 seed=0             # paper sanity baseline (~20 min on CPU)
uv run python -m q2jepa.train encoder.res=64 prior=mixture seed=0
uv run python -m q2jepa.train trainer.iters=20 readout.iters=20 eval.n_obs=4   # quick smoke run
uv run python -m q2jepa.train --cfg job                          # print resolved config, don't run
```

- Each run writes `ckpt.pt`, `metrics.json` and the resolved `.hydra/config.yaml` to its Hydra output dir
  (`outputs/<date>/<time>/` by default).
- **Resume** = rerun with the same output dir: `hydra.run.dir=outputs/my-run`. There is no
  `checkpoint_dir` key.
- `trainer.stop_after=N` ends a session after N iterations so long runs can be chunked across Colab sessions.
- `quantum=gpu` sets the `nvidia` target (Linux + NVIDIA GPU only); `trainer.device` defaults to `auto`.

### Benchmark

```bash
uv run python scripts/bench_attention.py --target qpp-cpu --res 16 --res 32 --images 16
```

## 5. End-to-end GPU loop on Colab (`colab` CLI)

**Scripted path:** `bash .agents/skills/colab-e2e/run.sh` runs steps 0–5 below, checks the output
notebook, and stops the VM even on failure (skill: [.agents/skills/colab-e2e/SKILL.md](.agents/skills/colab-e2e/SKILL.md)).
It rents a billable GPU, so run it only when the user asks. The manual steps below are for debugging.

Operating details (auth scopes, flags, recovery) are in the skill
[.agents/skills/colab-operator/SKILL.md](.agents/skills/colab-operator/SKILL.md) — read it before the first `colab` call.

**The notebook runs code from GitHub, not your working tree.** `notebooks/colab_gpu.ipynb` does
`git clone` of `REPO_URL` (default branch). Anything not committed and pushed is not tested. Push first,
or ask the human before pushing; to test a branch, change the clone cell to `git clone -b <branch>` and
don't commit that edit.

Steps:

```bash
git status && git push                                  # 0. Colab only sees pushed code
uv tool install google-colab-cli                        # 1. once
colab whoami                                            # 2. 401/403 → re-mint ADC scopes (see skill)
colab new -s q2jepa --gpu T4                            # 3. always name the session; 400 → no GPU quota
colab exec -s q2jepa --timeout 600 -f notebooks/colab_gpu.ipynb   # 4. writes notebooks/colab_gpu_output.ipynb
colab stop -s q2jepa                                    # 5. ALWAYS, also after failures
```

**Drive / checkpoints.** `colab drivemount` is interactive: an agent cannot run it. Either ask the human
to run `colab drivemount -s q2jepa` before step 4, or, for a throwaway test run, create the fallback dir:

```bash
echo 'import os; os.makedirs("/content/drive/MyDrive", exist_ok=True)' | colab exec -s q2jepa
```

With the fallback, checkpoints live on the VM disk and **are lost at `colab stop`** — fine for verifying
tests/benchmarks, not for training runs you intend to resume or report. Say which mode you used.

What the notebook does, in order: `nvidia-smi` → mount check → clone + `pip install -e .` →
`scripts/colab_setup.py` (swaps in the CUDA-12/13 `cuda-quantum` wheel matching the driver) →
quantum tests on `nvidia` → `bench_attention.py` on `qpp-cpu` and `nvidia` → a training run checkpointing
to `/content/drive/MyDrive/q2jepa/runs/<name>` (rerun the cell to resume).

Verify by reading `notebooks/colab_gpu_output.ipynb`: tests passed, both benchmark tables printed, and
the training cell printed a metrics dict. Don't commit `colab_gpu_output.ipynb` unless asked.

On failure: `colab log -s q2jepa -n 50` first. Wedged kernel → `colab restart-kernel -s q2jepa`.
Session gone (404/401 on exec) → `colab sessions`, then `colab new` again. Kernel state persists between
`colab exec` calls in one session, so you can rerun single steps with piped code instead of the whole notebook.

## 6. Definition of done

- `uv run pytest` and `uv run ruff check .` pass locally (Lefthook enforces this on commit).
- Changes to `src/q2jepa/`, `conf/`, `scripts/` or `notebooks/` get a spec-rules review before commit/PR
  (Claude Code: the `spec-rules-reviewer` subagent in `.claude/agents/`; other agents: apply its checklist).
- If you touched `src/q2jepa/quantum/`, the quantum tests also pass on the `nvidia` target via Colab
  (notebook section 1), or you state clearly that GPU verification was not run.
- New quantum kernels come with an equivalence test (e.g. factored vs. full swap test; CUDA-Q vs. Qiskit
  Born probabilities **including bit order** — Qiskit is little-endian).
- Reported results get a row in `docs/results/*.md` with overrides, seed, commit SHA and all metrics
  (coverage, validity, TV, minADE, ES), next to the matching classical baseline.
- Classical MovingMNIST acceptance: validity ≥ 85%, coverage ≥ 80%, TV ≤ 0.25.
- Never claim a Colab run succeeded without reading its output notebook.
