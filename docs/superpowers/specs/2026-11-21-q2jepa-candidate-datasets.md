# Q²-JEPA candidate datasets beyond MovingMNIST, HEP18 and D3IL

Status: proposal, not scheduled. Nothing here runs before the 27 Nov deadline.

## 1. Why

MovingMNIST, HEP18 and D3IL are rendered images with strong local structure, which favours a CNN.
We want benchmarks where the useful signal is long-range or relational, and where one context has
several valid futures (so coverage/validity/TV still apply).

Caveat: if attention beats the CNN there, that says nothing about HQViT vs. the ViT twin. The
headline comparison stays HQViT vs. `encoder=vit` under identical settings (AGENTS.md §3).

## 2. Candidates

| Dataset | Source | Size / license | Why it fits | Futures and metrics |
|---|---|---|---|---|
| Lichess Evaluations | `lichess_db_eval.jsonl.zst`, https://database.lichess.org | 417.9M positions (2026-11-11), CC0 | Piece interactions span the whole board; an 8×8 board is a natural 64-token input | Several good moves per position (multi-PV). Validity = sampled move is within a cp margin of the best PV; coverage = share of good PVs hit; TV over PV moves |
| NRI charged particles | `data/generate_dataset.py --simulation charged` in https://github.com/ethanfetaya/NRI (default `--n-balls 5`); Kipf et al., *Neural Relational Inference for Interacting Systems*, ICML 2018, arXiv:1802.04687 | Generated locally, any size | Every particle acts on every other one (no locality); can be rendered as images like HEP18 | Stochastic initial charges/velocities give several futures per context; minADE and ES on trajectories, validity from simulator energy/physics checks |
| Lichess Puzzles (secondary) | `lichess_db_puzzle.csv.zst`, same site | 6.19M puzzles, CC0 | Same board input as above | Usually one correct line, so weak for multi-future metrics; use only as a sanity task |
| Google Speech Commands (example only) | Warden, arXiv:1804.03209; `torchaudio.datasets.SPEECHCOMMANDS` (v0.02 default) | ~105k 1 s clips | Spectrograms have long-range time-frequency structure | Classification, not future prediction; poor fit, listed for completeness |

Recommended order: NRI charged (closest to HEP18, reuses the image renderer path), then Lichess Evaluations.

## 3. How one would be added

Per AGENTS.md §3, a new benchmark is a `data/` module + `benchmarks/` class + `readout/` module +
`conf/dataset/*.yaml` + `conf/experiment/*.yaml` + metrics. No new training script. It must be trained
and reported on images (or the board tensor as an image), every encoder at the same `encoder.res`,
tuning on `eval.split=dev`, with a classical baseline row next to each quantum row. Write a dataset spec
(acquisition, rendering, acceptance tests) like the HEP18/D3IL one before any code.

## 4. Built for fast iteration (2026-11-22)

Team decision (2026-11-22): try charged particles and chess quickly instead of the full HEP18 grid (stage D,
~42 GPU-hours on a table at the floor: CNN 0.3% validity). No per-benchmark tuning sweep: both reuse MovingMNIST's
model sizes and tuned schedules (`TUNED_ON` in `scripts/run_grid.py`). First pass: one seed, then more seeds only
where the result is informative.

| | Charged particles (`nri`) | Chess (`chess`) |
|---|---|---|
| Data | `data/nri.py`: own NRI-style simulator, generated on the fly | `scripts/fetch_chess.py`: first 10208 qualifying positions of the Lichess eval stream → `data/chess/positions.npz` |
| Futures | 4: hidden charges (+1, ±1, ±1) switch on at t = 0; balls fly freely while observed. Futures ≥ 5 px apart at t = 6 | 3: the top 3 engine moves, white to move, all within 30 cp of the best |
| Clip | 3 balls as Gaussian blobs of different widths, frames t = −2, −1, 0 / t = 2, 4, 6 | board: white pieces, black pieces, all pieces (8 px squares, intensity = piece type) |
| State, match | ball positions in the 3 future frames (18 numbers), balls matched as a set | the move's squares (4 numbers) |
| Splits | 20000 / 80 / 128 (one stream per split) | 10000 / 80 / 128 |

Shared code: `benchmarks/futures.py` (whole-scene clips, J futures as targets), `readout/futures.py` (MLP [h; t] →
state), `metrics.futures_metrics` (coverage, validity, TV vs uniform, minADE and ES as RMS distance in state units: px, squares) and a
readout-only validity diagnostic (§4.1). Runs: `python scripts/run_grid.py --bench <nri|chess> --stage grid --seeds 0`.

Chess is not a video world model: the "clip" is three views of one board and the futures are positions one move
later. It tests whether the latent prior can cover several equally good discrete outcomes; results tables say so.

### 4.1 Readout fix (2026-11-22)

The first particle grid (800 train scenes, 18 coordinates each within 2 px) gave validity 0 for CNN and ViT, and
`readout_validity` 0 too: the readout could not decode even the true futures. Readout diagnostic on the CNN run
(frozen encoder, dev split; RMS error per coordinate, nearest = decoded future is nearest to its own):

| Readout train data | Readout | Train RMS px | Dev RMS px | Dev nearest | Dev unique match |
|---|---|---|---|---|---|
| 800 scenes | MLP 128, 4000 steps | 3.2-4.2 | 6.2-10.4 | 0.60 | 0.07 |
| 800 scenes | MLP 512, 10000 steps, balls as a set | 0.1 | 5.8-7.3 | 0.85 | 0.16 |
| 20800 scenes | MLP 512, 10000 steps, balls as a set | 2.9-3.2 | 3.4-3.7 | 0.98 | 0.62 |

The readout memorised 800 scenes. Changes: particles train on 20000 generated scenes, chess on 10000 positions
(sources kept compact on the device and rendered per batch); readout MLP 512 wide, 10000 steps of 256
(`conf/experiment/nri.yaml`, `chess.yaml`); balls matched as a set (the readout confuses them, they differ only in
width). Validity (`metrics.match_futures`): a sample is valid if it lies within half the gap from its nearest future
to that future's nearest other future, so it matches exactly one future, capped at 4 px (particles; about the widest
blob) and at the exact squares (chess: RMS 0.25). With balls matched as a set, particle ES is descriptive only. The
first-pass runs (800 scenes, old metric) are void: they were moved out of `runs/nri/grid` on the VM, so the grid
launcher cannot skip or resume them.

### 4.2 Seed-0 grid and second fix (2026-11-23)

Seed 0 on the fixed readout (commit 9c56c5a, test split, 128 observations): every row at the floor.

| Bench | Encoder | validity (QCBM / Softmax) | minADE | readout_validity |
|---|---|---|---|---|
| particles | CNN | 0.011 / 0.004 | 5.7 px | 0.42 / 0.39 |
| particles | ViT | 0.004 / 0.000 | 6.8 px | 0.02 |
| particles | HQViT | 0.002 / 0.000 | 6.8 px | 0.03 |
| chess | CNN | 0.025 / 0.008 | 1.5 sq | 0.01 |
| chess | ViT | 0.006 / 0.017 | 0.92 sq | 0.16-0.18 |
| chess | HQViT | 0.014 / 0.014 | 1.0 sq | 0.08-0.13 |

Two bottlenecks, two changes (team decision 2026-11-23):

- Chess: regressing 4 square coordinates to within 0.25 rarely hits the exact move, even from the true future. The
  readout now classifies the from- and to-square (2 x 64 logits, cross-entropy) and decodes the argmax squares
  (`benchmarks/chess.py`, `readout/futures.py`); validity is then an exact move match.
- Particles: ViT and HQViT mean-pool the patches and lose where each ball is (readout 2-3% vs CNN 42%).
  `conf/experiment/nri.yaml` sets `encoder.pool: flatten` for both (pooling is after the circuits, so the twin rule
  holds); the grid still uses MovingMNIST's mean-pooled schedules. `scripts/run_grid.py` reads the preset's pooling.
  The flattened fusion head takes 3 x 64 x 64 inputs (CNN: far fewer) on an untuned schedule, so a particle
  ViT-vs-CNN gap is not an encoder effect alone; HQViT vs ViT stays matched.

The seed-0 runs above are void for the new grid: move `runs/nri/grid` ViT/HQViT runs to `runs/nri/grid-meanpool/`
(the launcher refuses them otherwise) and all of `runs/chess/grid` to `runs/void/chess-regression/` before rerunning:
the launcher skips any run with `metrics.json` and would mix the two chess readouts. The particle CNN runs are
unchanged by this fix and can stay.
