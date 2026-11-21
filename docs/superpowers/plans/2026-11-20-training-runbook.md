# Training runbook (2026-11-20)

How to run the phase-2 grid (spec §9) on Colab T4s with `scripts/run_grid.py`. The rules are in `AGENTS.md` §3
and spec §4.

## 0. Before the grid

| Step | Status |
|---|---|
| GPU quantum tests on `nvidia`, including the HEP18 grid circuit | passed (T4, 51/51) |
| HEP18 render check: readout-only validity ≥ 90% for CNN **and** twin at σ = 1.25 | CNN 97.9% passes; twin 31.8% fails (untuned) |
| HEP18 world model: CNN on images 0.1–0.4% validity (7 runs), state encoder 44% (paper: 98%) | §1a done (8f74d5f): no setting fixes it; `cnn-lre-lo` wins, no config change; C may start |
| HEP18 wrapped ("snake") layout, `dataset.layout=snake` | rejected: CNN 0% validity, twin readout 0.5% |
| D3IL data: `data/d3il/avoiding.npz` (`scripts/fetch_d3il.py`) | missing |
| D3IL render check: same rule as HEP18, fold-0 train demos only | after the data |
| Twin/HQViT pooling switched to soft-argmax (spec §5.6): move `mmnist/tune/{vit,hqvit}-s*` to `mmnist/tune-meanpool/` and `hep18/fix/vit-*` to `hep18/fix-meanpool/`, then rerun A and the twin part of §1a; `run_grid.py` refuses stale tune runs | moved; twin part of §1a rerun; A running |

If the D3IL render check fails on the twin, stop and take the numbers to the team (dataset spec §8). HEP18 never
stops: §1a picks its best setting and the grid runs with it.

## 1. Who runs what

Each person uses their own Colab account; one table never mixes accounts (F9).

| Person | Work, in order | GPU-hours | Starts |
|---|---|---|---|
| Daria | A + B (MovingMNIST, §3); HEP18 fix sweep (§1a); C + D (HEP18) | ~16 + ~2 + ~67 | A now; the sweep after A (one T4 takes 3 runs at once); C once her §1a PR is merged |
| Paul | D3IL data + render check; E (D3IL) | ~1 + ~70 | data now; E once the data is in and Daria's A is finished |

D3IL reuses MovingMNIST's tuning (A), which lives on Daria's Drive. Daria shares her `q2jepa/runs` folder with Paul. Paul
adds a shortcut to it in his My Drive and names it `q2jepa-daria`. E then reads A's results from there with
`--tune-root`.

**Progress, visible to the team (no paid service):**
- Share the Drive folder `q2jepa/runs` with the team as viewer.
- `<root>/<bench>/<stage>/progress.md` is rewritten every 5 minutes. It shows each run's status (done, running,
  failed, pending), its latest step and loss, and the headline metrics of finished runs.
- Each run logs every loss to `<run dir>/log.jsonl`.
- **Results:** when a stage ends, run the printed `scripts/plot_results.py` command. Then open a PR adding the
  table and figure to `docs/results/`, with the SHA and the account's GPU type.

## 1a. HEP18 fix sweep (Daria, ~2 h on one T4)

Not a results table, so it has its own script instead of the launcher. Dev split. Setup as in §2.

All runs are on rendered images.

- **CNN (3 runs):** the prior: mixture prior, prior learning rate 3e-3, 12000 iterations.
- **CNN and twin (7 runs):** the encoder learning rate (1e-3, 1e-4) and tuning schedules s2 and s3.

```python
!python scripts/hep18_fix_sweep.py --jobs 3   # writes to MyDrive/q2jepa/runs/hep18/fix, prints a table at the end
```

After a disconnect, rerun the same cell: finished runs are skipped. `--dry-run` prints the commands.
The `vit-*` runs must use this commit's pooling (`encoder.pool` in their `.hydra/config.yaml`); the sweep script does not check it.

**What to decide (Daria owns the HEP18 fix; no team sign-off needed):**
- Pick the best setting: the highest dev validity among settings where the CNN and the twin both have
  `readout_validity` ≥ 0.9. If no setting passes that check, pick the highest dev validity anyway.
- Where the winner goes (PR with the sweep table in the description; C starts after the merge):

  | Winner | Change |
  |---|---|
  | `cnn-lrp` or `cnn-long` | `trainer.lr_prior` or `trainer.iters` in `conf/experiment/hep18.yaml` |
  | an encoder learning rate or schedule (`*-lre-*`, `*-t2`, `*-t3`) | none: C tunes s1–s3 itself (`+tune` sets the encoder learning rate) |
  | `cnn-mix` | none: the grid priors stay fixed (spec §9); note the mixture number in the results |
  | no run beats the defaults | none |
- HEP18 is never dropped. If nothing gets close to the paper's 98%, run C + D with the best setting and report
  what it reaches. The HEP18 results table states the gap to the paper and the sweep that picked the setting.

**Result (2026-11-21, commit 8f74d5f, seed 0, dev split, one Colab T4; twin runs with soft-argmax pooling):**

| Run | Validity | Coverage | Readout validity | TV |
|---|---|---|---|---|
| **cnn-lre-lo** | **0.0032** | **0.0104** | 0.990 | 0.997 |
| cnn-lrp | 0.0032 | 0.0069 | 0.977 | 0.997 |
| cnn-t3 | 0.0022 | 0.0052 | 0.997 | 0.998 |
| cnn-t2 | 0.0015 | 0.0061 | 1.000 | 0.999 |
| cnn-mix | 0.0015 | 0.0069 | 0.984 | 0.999 |
| cnn-lre-hi | 0.0013 | 0.0026 | 0.880 | 0.999 |
| cnn-long | 0.0011 | 0.0035 | 0.987 | 0.999 |
| vit-lre-hi | 0.0001 | 0.0009 | 0.254 | 1.000 |
| vit-t2 | 0.0000 | 0.0000 | 0.676 | 1.000 |
| vit-t3 | 0.0000 | 0.0000 | 0.479 | 1.000 |

- No setting passes the readout check: the twin reaches at most 0.676.
- Highest validity: `cnn-lre-lo` and `cnn-lrp` tie at 0.0032. Coverage breaks the tie (as in F5), so `cnn-lre-lo` wins.
  It is an encoder learning rate, so nothing changes: C tunes s1–s3 itself.
- The sweep does not fix HEP18: every setting is within noise of the defaults, far from the paper's 98%. C + D run
  with the defaults and the results table reports the gap.
- The twin does not recover the state from HEP18 images even with soft-argmax (readout 0.25–0.68 vs CNN ~0.99).

## 2. Setup (one Colab T4 session)

```python
!git clone https://github.com/Brigada-Inginerilor-Amarati/Q2-JEPA /content/Q2-JEPA
%cd /content/Q2-JEPA
!pip install -q -e . && python scripts/colab_setup.py
!git rev-parse HEAD        # record it: every results row needs the SHA
from google.colab import drive; drive.mount("/content/drive")   # runs live on Drive
```

D3IL also needs `data/d3il/avoiding.npz` copied into `/content/Q2-JEPA/data/d3il/`.

- **One table, one account.** Every run of one benchmark runs on the same Colab account and GPU type (F9). Different
  benchmarks may run on different accounts.
- Runs go to `/content/drive/MyDrive/q2jepa/runs/<bench>/<stage>/<run>` (`--root` changes it).
- **Resume:** after a disconnect, rerun the same cell. Finished runs are skipped, interrupted ones resume from their
  checkpoint.
- The launcher adds `quantum=gpu trainer.device=cuda` and, for HQViT, `quantum.hqvit_train=torch`.

## 3. Commands

```python
!python scripts/run_grid.py --bench mmnist --stage tune --jobs 3   # A. 9 runs, dev split
!python scripts/run_grid.py --bench mmnist --stage grid --jobs 3   # B. 18 runs, test split
!python scripts/run_grid.py --bench hep18 --stage tune --jobs 3    # C. 9 runs
!python scripts/run_grid.py --bench hep18 --stage grid --jobs 3    # D. 21 runs (+ Product-QCBM for HQViT)
!python scripts/run_grid.py --bench d3il --stage grid --jobs 3 \
    --tune-root "/content/drive/MyDrive/q2jepa-daria"                 # E. 54 runs (3 folds); needs A finished
```

- `tune` runs every encoder with schedules s1–s3, seed 0, Softmax-Categorical, on the dev split.
- `grid` picks each encoder's best schedule from its tune runs: best dev coverage, ties broken by lower TV. D3IL
  has no dev split and reuses MovingMNIST's. Each grid stage refuses to start until its tuning is finished.
- `grid` runs HQViT vs twin with Entangled-QCBM first.
- `--dry-run` prints the commands without running anything. `--encoders` and `--seeds` restrict a stage.
- When a stage ends, it prints the `scripts/plot_results.py` command for its table and figure.

To look at single runs, use `scripts/show_futures.py` (MovingMNIST), `scripts/show_hep18.py` or
`scripts/show_d3il.py` with `<run dir>`.

## 4. Priority if time runs out

1. A + B (MovingMNIST)
2. C + D, HQViT vs twin with `entangled_qcbm`
3. E, HQViT vs twin with `entangled_qcbm`
4. Everything else

## 5. Cost per run on a T4 (measured 2026-11-20, `scripts/bench_encoder_step.py`)

Hours per run, Softmax-Categorical / Entangled-QCBM prior. Evaluation is not included. It is small: a full HQViT
HEP18 run of 40 steps plus the CUDA-Q evaluation took under 2 minutes.

| Benchmark | CNN | twin | HQViT (torch) |
|---|---|---|---|
| MovingMNIST | 0.02 / 0.22 | 0.05 / 0.22 | 1.6 / 1.7 |
| HEP18 | 0.16 / 1.4 | 0.28 / 1.9 | 3.8 / 5.3 |
| D3IL | not measured (no data) | | |

HEP18 subtracts the readout term the script over-counts (the readout trains on cached features).

Run-hours on one GPU:

| Part | Hours |
|---|---|
| MovingMNIST tuning + grid | ~16 |
| HEP18 tuning + grid + Product row | ~67 |
| D3IL, all 3 folds (guess) | ~70 |
| **Total** | **~150** |

`--jobs 3` on one T4 should bring the wall-clock time to about 3–4 days; one account per benchmark cuts it to about 1–2. A Colab session costs ~1.15
units per hour, whatever runs in it.

## 6. Reporting

One table per benchmark in `docs/results/`:
- each row has overrides, seed, SHA, platform, parameter counts (quantum / classical) and all metrics;
- mean ± s.d. over seeds;
- HQViT rows say "trained in torch, evaluated in CUDA-Q".

If two rows are within one s.d., add seeds 3–4 before concluding (F6): `--seeds 3 4`. Test metrics are
`metrics.json`, dev metrics are `metrics_dev.json`.
