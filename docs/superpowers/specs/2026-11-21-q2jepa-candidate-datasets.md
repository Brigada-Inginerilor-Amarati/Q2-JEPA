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
