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
| [docs/superpowers/specs/2026-11-03-q2jepa-phase2-design.md](docs/superpowers/specs/2026-11-03-q2jepa-phase2-design.md) | Phase 2: fairness protocol, ViT twin, HQViT/QCBM design, HEP18 + D3IL as images, grid, build order. Overrides the kickstart spec where they differ |
| [docs/superpowers/specs/2026-11-03-q2jepa-datasets-hep18-d3il.md](docs/superpowers/specs/2026-11-03-q2jepa-datasets-hep18-d3il.md) | HEP18 + D3IL data, acquisition, rendering, interfaces and acceptance tests |
| [docs/superpowers/plans/2026-10-28-q2jepa-kickstart.md](docs/superpowers/plans/2026-10-28-q2jepa-kickstart.md) | Kickstart tasks 1–11 (done) and their acceptance criteria |
| [docs/results/](docs/results/) | Reported numbers; every row reproducible from overrides + seed + SHA |

## 2. Current state vs. target

The README describes the target. What actually exists today:

| Part | Status |
|---|---|
| Stochastic MovingMNIST (`data/mmnist.py`) | done |
| CNN encoder (`encoder/cnn.py`, `encoder=cnn`) | done |
| Classical ViT twin of HQViT (`encoder/vit.py`, `encoder=vit`) | done; defaults 8×8 patches + sinusoidal positions + mean pooling (soft-argmax tried and rejected; phase-2 spec §5.3, §5.6) |
| Classical priors: Softmax-Categorical, Mixture (`prior/classical.py`) | done |
| Predictor, losses, readout, metrics, 3-stage `train.py` with resume | done |
| Swap-test attention kernels, factored 4q + full 2n+1 q (`quantum/attention.py`) | done; used for equivalence tests and the benchmark |
| HQViT encoder (`encoder/hqvit.py`, `quantum/hqvit.py`, `encoder=hqvit`; inherits `vit.yaml`) | built; **not trained yet**. CUDA-Q training: ~25 h/run on a T4. `quantum.hqvit_train=torch` (exact torch simulation, evaluated in CUDA-Q): 1.6 h/run (MovingMNIST) on a T4 |
| QCBM priors: entangled / product / frozen (`prior/qcbm.py`, `quantum/bridge.py`, `prior=*_qcbm`); parameter shift + ideal reference (`quantum.grad`) | done; verified on `nvidia` T4 in fp64 (2026-11-20); `quantum.precision` selects fp32/fp64 |
| IBM inference (`scripts/eval_hardware.py`, `[ibm]` extra) | **not built** |
| Dev-split evaluation (`eval.split=dev` → `metrics_dev.json`) | done |
| Checkpoint `evaluate()`, Dice, readout-only diagnostic | **not built** |
| HEP18 and D3IL data and renderers (`data/hep18.py`, `data/d3il.py`, `scripts/fetch_d3il.py`, `conf/dataset/`) | done; σ provisional until the render check (dataset spec §8) |
| HEP18 and D3IL training (`benchmarks/`, `readout/hep18.py`, `readout/d3il.py`, `metrics.py`, `+experiment=hep18`, `+experiment=d3il`) | built. HEP18 under-trains (2026-11-20, dev: CNN on images 0.1–0.4% validity, `encoder=state` 44%, paper 98%); Daria's fix sweep in the runbook §1a picks the best image setting; HEP18 is then reported with it, whatever it reaches (never dropped). `dataset.layout=snake` tried and rejected. D3IL needs `data/d3il/avoiding.npz` |
| Charged particles and chess, fast iteration (`data/nri.py`, `data/chess.py`, `scripts/fetch_chess.py`, `benchmarks/futures.py`, `+experiment=nri`, `+experiment=chess`) | built 2026-11-22; no own tuning (MovingMNIST's schedules), one seed first (candidate datasets spec §4). Readout fix 2026-11-22 (§4.1): 20k/10k train scenes, set-matched validity; first-pass runs void. HEP18 stage D replaced by this |
| Grid launcher and plots (`scripts/run_grid.py`, `scripts/plot_results.py`, `scripts/show_hep18.py`, `scripts/show_d3il.py`) | done |

Do not launch full `encoder=hqvit` training with the default `quantum.hqvit_train=cudaq` (~25 h/run on a T4). Next build order is
phase-2 spec §10: evaluation refactor → twin positional experiment → QCBM prior → T4 timing gate → HQViT →
MovingMNIST grid → IBM inference → HEP18 → D3IL. Plan new work with a spec/plan under `docs/superpowers/`.

## 3. Hard rules

From the spec; breaking any of these makes results invalid even if the code runs.

- `cudaq` is imported **only** under `src/q2jepa/quantum/`. Everything else calls into it.
- `qiskit` is imported **only** in `scripts/eval_hardware.py` and its test.
  (Both enforced by `tests/test_import_boundaries.py`.)
- Every reported number is computed through CUDA-Q. One exception: HQViT may *train* with
  `quantum.hqvit_train=torch`, an exact torch simulation of the same circuit (`quantum/torch_sim.py`, checked
  against CUDA-Q by `tests/test_torch_sim.py`). Evaluation always runs the CUDA-Q circuits. Results tables say
  which simulator trained each HQViT row. No other bypass.
- No quantum result without a classical baseline trained under identical settings (same `encoder.res`, seed, trainer).
  Only the component under test changes between compared rows.
- HQViT claims are made against the classical twin (`encoder=vit`) with the same patching, blocks, FFN,
  positional handling, pooling and fusion; the CNN is an anchor, not the headline baseline.
- Every encoder in the grid, CNN included, sees the same input resolution (`encoder.res`) with the same
  `encoder.input_norm` (`l2` whenever HQViT is compared — `+experiment=phase2` sets that and native 64×64); data, readout and metrics stay at native
  resolution / state space (64×64 for MovingMNIST; metres and routes for HEP18 and D3IL).
- All rows of one comparison table run on the same platform; numbers differ across machines.
- Tuning and model selection use `eval.split=dev` (the `+tune=s1|s2|s3` presets set it); the test split is
  evaluated once per final checkpoint. ViT/HQViT rows always use a preset: `+experiment=phase2` for MovingMNIST (without it `res` is 16),
  `+experiment=hep18` or `+experiment=d3il` for those benchmarks (never stacked with `phase2`).
- **All training runs on a Colab GPU** (`quantum=gpu trainer.device=cuda`): tuning, grid, sweeps, debug and
  smoke runs alike. Never train on a local CPU. Locally you only run `pytest`, `ruff` and `--cfg job`.
- Tuning and grid runs go through `scripts/run_grid.py` (§4 "Grid runs"), never hand-written commands.
- HEP18 and D3IL runs train and report on rendered images. `encoder=state` is a diagnostic only: never in a
  sweep, tuning, grid or results table (`scripts/plot_results.py` refuses it).
- Every reported number must be regenerable from config overrides + seed + commit SHA. No manual
  notebook tweaks: change config or code, not the notebook.
- A new benchmark = a `data/` module + a `benchmarks/` class + a `readout/` module + `conf/dataset/*.yaml` +
  `conf/experiment/*.yaml` + metrics. Never a new training script.
- No hardware training; IBM QPUs are inference-only on the trained Born prior.
- The notebook stays thin: it installs the package and calls into it. No logic lives there.
- Docs, specs, plans, results, comments and commit messages are in **English**. Keep them short and plain:
  facts, tables and commands; no padding, no restating other docs.

## 4. Local dev loop (tests only) and training on GPU

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

Training runs on a Colab GPU only (§3, §5). Locally, only check the resolved config:

```bash
uv run python -m q2jepa.train +experiment=hep18 encoder=cnn --cfg job   # print resolved config, don't run
```

On the Colab VM (`colab exec`, or the runbook's notebook cells):

```bash
python -m q2jepa.train encoder.res=64 seed=0 quantum=gpu trainer.device=cuda      # paper sanity baseline
python -m q2jepa.train trainer.iters=20 readout.iters=20 eval.n_obs=4 quantum=gpu trainer.device=cuda   # smoke
```

- Each run writes `ckpt.pt`, `metrics.json` and the resolved `.hydra/config.yaml` to its Hydra output dir
  (`outputs/<date>/<time>/` by default).
- **Resume** = rerun with the same output dir: `hydra.run.dir=outputs/my-run`. There is no
  `checkpoint_dir` key.
- `trainer.stop_after=N` ends a session after N iterations so long runs can be chunked across Colab sessions.
- `quantum=gpu` sets the `nvidia` target (Linux + NVIDIA GPU only); `trainer.device` defaults to `auto`.

### Grid runs: always `scripts/run_grid.py`

Tuning and grid runs (anything that feeds a results table) go through the launcher, on a Colab T4. Never
hand-write `q2jepa.train` commands for them. The launcher has no CPU mode. Runbook:
[docs/superpowers/plans/2026-11-20-training-runbook.md](docs/superpowers/plans/2026-11-20-training-runbook.md).

```bash
python scripts/run_grid.py --bench <mmnist|hep18|d3il> --stage <tune|grid> --jobs 3 [--dry-run]
```

It sets the preset, the GPU platform, `quantum.hqvit_train=torch` for HQViT, the tuned schedule per encoder,
seeds and folds. It writes one dir per run under `--root`, skips finished runs and resumes interrupted ones,
and keeps `<root>/<bench>/<stage>/progress.md` current from each run's `log.jsonl`. Change
the grid in the script, not by hand. Results and figures: `scripts/plot_results.py`; single runs:
`scripts/show_futures.py` (MovingMNIST), `scripts/show_hep18.py`, `scripts/show_d3il.py`.

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
