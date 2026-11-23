# Q²-JEPA "doors": a synthetic multi-future benchmark (proposal, owner: Paul)

Status: proposal, 2026-11-23. Team decision: build it if it can run before 27 Nov; otherwise report without it.

## 1. Why

Chess was the only fast-iteration benchmark to leave the floor, and only after its readout became a classification
of squares (candidate-datasets spec §4.2). Doors keeps the parts that worked: a small set of equally likely
discrete futures, visible in the image, decoded by classification. It drops what hurt particles (continuous
positions matched to within a few px) and chess (no motion, sparse 3k-move data). Generated locally, any size.

## 2. Data

| | |
|---|---|
| Grid | 8 x 8 cells, 8 px each, rendered 64 x 64 (as chess) |
| Scene | an agent (bright 6 x 6 blob) in a room; D = 6 candidate door cells on the outer wall, J = 3 of them open (drawn as gaps), the rest wall |
| Observed clip | 3 frames: the agent walks 2 cells towards the room centre (gives motion, no hint of the choice) |
| Futures | J = 3, equally likely: the agent steps to each open door along a shortest path; future clip = its cells at 3 equally spaced steps (last frame on the door) |
| State | (agent file, rank at t = 0, door file, rank): 4 numbers, the same shape as chess, so `benchmarks/chess.py`'s square readout (2 x 64 logits) and `metrics.futures_metrics` with `max_radius = 0.25` apply unchanged |
| Splits | train 20000, dev 80, test 128, one RNG stream per split (as `data/nri.py`) |
| Redraw | scenes where two open doors share a cell or the path is blocked |

Validity is an exact door match; coverage = share of the 3 open doors hit; TV vs uniform over them. Chance levels
to report next to results: a uniformly random open door is always valid; a random wall cell is valid 3/28 of the time.

## 3. Code (AGENTS.md §3: no new training script)

- `data/doors.py`: `generate(n, seed)` -> obs (n, 3, 2), fut (n, 3, 3, 2) agent cells, doors (n, D), open mask;
  `render(cells, doors, open)` -> (..., 64, 64) on the device.
- `benchmarks/doors.py`: `FuturesBenchmark` subclass; copy `readout_dim`, `readout_fit_loss`, `decode` from chess.
- `conf/dataset/doors.yaml`, `conf/experiment/doors.yaml` (copy `chess.yaml`; `input_scale` from a test like
  `test_input_scale`); add `doors` to `PRESET` and `TUNED_ON` (`"mmnist"` first) in `scripts/run_grid.py`.
- Tests: futures are J distinct open doors; render puts the agent where the state says; square readout round trip;
  a tiny end-to-end smoke run like `test_chess_smoke`.

## 4. Runs

Seed 0 grid first (`run_grid.py --bench doors --stage grid --seeds 0`), then seeds 1-2 only if off the floor.
Check `readout_validity` first: below ~0.5 for every encoder means the readout, not the model, is the limit.
