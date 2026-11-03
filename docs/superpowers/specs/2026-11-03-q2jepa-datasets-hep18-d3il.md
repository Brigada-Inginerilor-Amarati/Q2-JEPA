# Q²-JEPA — Dataset Spec: HEP18 and D3IL as Rendered Images

Date: 2026-11-03 · Status: ready for implementation · Parent: [phase-2 design](2026-11-03-q2jepa-phase2-design.md)
§3, §8.3, §8.4 (this document is the detailed version; where they differ, this one wins for data and rendering).

## 1. Goal and scope

Build the two new benchmarks so that **every encoder (CNN, classical ViT twin, HQViT) can consume them as images**,
while the readout and metrics stay in the Q-JEPA paper's native state space.

In scope (this spec):
- data generation / loading, splits and preprocessing, exactly as in the Q-JEPA paper (`docs/papers/qjepa.pdf`);
- **rendering** states into 64×64 image clips;
- the dataset interface the training code will use, and the tests that prove it correct.

Out of scope (done separately in the main pipeline): readouts, metrics, model changes (agent-local bits, change
regularization, joint energy, recursive rollout loop). This spec only fixes the fields those need (§4.6, §5.6).

Why images at all: in the paper neither task uses images (HEP18 feeds six scalars per car to an MLP; D3IL feeds 24
trajectory points to a GRU). HQViT needs images. We render the paper's states; the paper's numbers become a
reference, and our own CNN on the same renders is the baseline.

## 2. Ground rules (from the repo's hard rules — please read `AGENTS.md` §3)

1. **Only the encoder input is an image.** Everything a readout or metric needs is returned as state tensors in the
   paper's units (metres, m/s, normalized coordinates). Never decode positions back from images.
2. **Native 64×64, no downsampling.** Renderers output exactly `(…, 3, 64, 64)` float32 in [0, 1]. The pipeline
   applies its own normalization afterwards (`encoder.input_norm`, `encoder.input_scale`); do not normalize.
3. **Deterministic.** Everything is a pure function of `(split, seed)` (and fold for D3IL). Use
   `np.random.default_rng([seed, split_id, ...])`; never global RNG state, never time-based seeds.
4. **Renderers are pure torch functions of states, batched, device-agnostic.** D3IL re-renders model predictions
   inside evaluation (up to 16 recursive calls × 384 rollouts), so rendering must accept a batch tensor on CPU or
   GPU and must not loop in Python over samples.
5. **Sub-pixel information must survive.** Draw with anti-aliased Gaussian splats (or analytically area-weighted
   lines), never by rounding to integer pixels. Tests in §6 check this.
6. **No static background.** Don't draw roads, lanes, obstacles or borders: they carry no information about the
   sample and dilute amplitude encoding (HQViT normalizes the whole clip).
7. **A new benchmark = `data/` module + `readout/` module + `conf/dataset/*.yaml` + metrics.** No new training
   script. You own `data/` and the dataset yaml; readout/metrics are ours.
8. **Downloaded data is untrusted** (§5.1): download into its own directory, never `pickle.load` or `torch.load`
   it in the main process before inspecting it, and convert once to a safe format (`.npz`, plain arrays).

## 3. Common interface

Follow `src/q2jepa/data/mmnist.py` (`StochasticMovingMNIST`): a `torch.utils.data.Dataset` whose items are dicts.

- Constructor: `Dataset(split: Literal["train", "dev", "test"], root: str = "data", seed: int = 0, ...)`.
  The paper's validation split is our `"dev"` (used for tuning/selection; `"test"` is touched once per checkpoint).
- Items hold **states**; images are rendered on access by the module's `render_*` function (so the dataset stays
  small and the same renderer is used for model predictions). Cache rendered uint8 frames only if profiling says so.
- Each module exposes its renderer at module level, e.g. `render_agent_clip(states) -> Tensor`, with a docstring
  giving the world→pixel mapping.
- `conf/dataset/<name>.yaml` sets `name:` plus any sizes; add needed keys to `DatasetCfg` in `src/q2jepa/config.py`
  (unknown keys are rejected by the schema).
- Report the **mean L2 norm of a rendered 64×64 train frame** (over all frames of the train split) in the module
  docstring. We use it to set `encoder.input_scale = mean_norm / 64` for that dataset (MovingMNIST: 9.24 → 0.144).

## 4. HEP18 — HighwayEnv priority intersection (Q-JEPA §5.3, App. F.3, F.4, Table 5)

### 4.1 What it is
Four cars approach a straight-lanes intersection from north, west, south and east. Each observation has **18
equally likely joint futures**: one axis (NS or EW) has priority (p = ½); each priority car independently picks one
of three acceleration schedules; both yielding cars brake to a stop. 2 × 3² = 18. There is no released code or data
— we regenerate it from App. F.3.

### 4.2 Geometry
- Use HighwayEnv's straight intersection lanes (Leurent 2018, `highway-env`). Install it **only to read the
  geometry once** (lane width, lane centre offsets for each approach, driving side, vehicle length/width), then
  hardcode those constants in `data/hep18.py` with a comment citing the `highway-env` version and source lines.
  Do not make `highway-env` a runtime dependency; we don't use its driver policy.
- Every car drives **straight through** on its own approach lane (no turns).
- Per car keep a route frame: longitudinal coordinate `s` along its lane (s = 0 at the intersection centre
  line, increasing in the driving direction, so approaching cars have s < 0), plus heading φ.

### 4.3 Sampling one observation
For each car independently: initial speed `v0 ~ U[8, 9]` m/s and distance from the centre `d0 ~ U[43, 47]` m.

**Interpretation to confirm (inferred, not stated):** `d0` is the distance **at t = 0** (end of observation), and
the observation covers t = −2, −1, 0 s at constant speed `v0`, i.e. `s(t) = −d0 + v0 t` for t ∈ [−2, 0]. Reason:
with this reading the yielding cars stop 16–23 m before the centre ("let yielding vehicles stop before the
intersection"); measuring `d0` at t = −2 instead would stop them between 7 m before and 2 m past the centre,
i.e. inside the intersection.
Record the choice in the module docstring.

### 4.4 Futures
Three stages of 3 s (t ∈ [0, 3], [3, 6], [6, 9]); constant acceleration within a stage, integrated **exactly**
(`s += v·Δt + ½aΔt²`, `v += aΔt`). Schedules, accelerations per stage in m/s² (Eq. 17):
- A = (−0.5, 0, 0.5), B = (0, 0, 0), C = (0.5, 0, 0) — priority cars, each independently, p = ⅓ each;
- D = (−v0/5, −v0/10, −v0/30) — both yielding cars (speeds become 0.4 v0, 0.1 v0, 0).

Joint future index: `j = 9·axis + 3·a_first + a_second` with axis ∈ {0: NS priority, 1: EW priority} and the two
priority cars' schedules in a fixed order (document the order). All 18 futures exist for every observation.

**Validity check during construction:** step every trajectory at 0.05 s over [−2, 9] s and check vehicle rectangles
(HighwayEnv's vehicle length × width, oriented by heading) for overlap. If any of the 18 futures of an observation
collides, **resample that observation** (count and report how many were resampled; expect few or none).

### 4.5 Dataset size and splits
448 observations, split **by observation** (all 18 futures stay together): **256 train, 64 dev, 128 test**.
Generate all 448 from one seeded stream, then split, so the split is deterministic and disjoint.

### 4.6 States returned (units: metres, m/s, seconds)
Per item (one observation):

| Key | Shape | Content |
|---|---|---|
| `obs_state` | (4, 3, 2) | per car, times (−2, −1, 0): (s, v) longitudinal position and speed |
| `fut_state` | (18, 4, 3, 2) | per future, per car, times (3, 6, 9): (s, v) |
| `obs_xy` | (4, 3, 6) | the paper's stored state (x/50, y/50, vx/10, vy/10, cos φ, sin φ) — for the paper-encoder sanity row |
| `fut_xy` | (18, 4, 3, 6) | same for futures |
| `future_id` | (18,) | joint index j (evaluation only; never a model input) |
| `obs_idx` | () | observation index |

Car order everywhere: **N, W, S, E** (this is also the QCBM's qubit-row order, App. C).
For training, the pipeline uses all 18 futures per observation (Table 5: 18 target views per observation).

### 4.7 Rendering (new)
**One clip per car**, so the per-car encoder sees only its own motion — exactly the paper's per-agent encoder,
which also sees only that car's six scalars. Each car is drawn in its **own route frame, canonically oriented**
(the car always moves "up" the image), so one shared encoder serves all four cars, like the paper's shared MLP on
route-projected states.

- `render_agent_clip(s: Tensor (..., 3)) -> Tensor (..., 3, 64, 64)`: frame f draws one Gaussian blob at the
  car's position s[..., f]. Same function for observations (t = −2, −1, 0) and futures (t = 3, 6, 9).
- Mapping: row coordinate `r = (s − s_min) / (s_max − s_min) · 63` (top/bottom orientation: document it), column
  fixed at the image centre (31.5). Choose `[s_min, s_max]` to contain every position in the dataset with a
  margin: from the bounds of §4.3–4.4, obs s ∈ [−65, −43] and futures s ∈ [−30.2, +49.2] (yielding cars stop at
  s ∈ [−23, −16]); **compute the exact range from the generated data**, pad by ≥ 3σ of the blob, and hardcode it.
  That gives ~1.9 m/pixel.
- Blob: isotropic Gaussian, σ ≈ 1.0–1.5 px (tune with the test in §6), peak normalized to 1, evaluated
  analytically per pixel (no rounding). The blob's centroid must track `s` with sub-pixel accuracy: speed has to be
  readable from 1 s displacements (~8–9 m ≈ 5 px) to well under 1 m/s (≈ 0.5 px).
- Return shape for a full observation: obs clips (4, 3, 64, 64); futures (18, 4, 3, 64, 64).
- Open, decide with us after the first CNN readout-only check: σ, whether to use the second image axis (e.g. a
  speed bar), and whether the window should be cropped tighter.

### 4.8 Config
`conf/dataset/hep18.yaml`: `name: hep18`, `n_train: 256`, `n_dev: 64`, `n_test: 128`.

## 5. D3IL Avoiding (Q-JEPA §5.4, App. B)

### 5.1 Acquisition (untrusted data)
- Official source: [ALRhub/d3il](https://github.com/ALRhub/d3il) README → Google Drive zip
  (`https://drive.google.com/file/d/1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8/view`) containing all seven tasks. Do not
  use unofficial mirrors (e.g. the Hugging Face `krishpop/d3il_avoiding` copy) for reported data.
- Write `scripts/fetch_d3il.py`: download with `gdown` into `data/d3il/raw/` (its own empty directory; `data/` is
  git-ignored), record the zip's SHA-256 in the module docstring, extract **only the Avoiding task**.
- **Check the licence** of the data (the code is MIT; the README doesn't say for the data). Report before we use it.
- Inspect the file format **before loading it in Python**. If it is pickle / `torch.save`, convert it once in a
  separate, isolated process (`python -I`, run from a different directory) to `data/d3il/avoiding.npz` with plain
  float arrays, after reading what the pickle contains. Everything else loads only the `.npz`.
- Report: is the Avoiding data state-only, or does it include camera images? If images exist, tell us before
  building the renderer — real frames may be preferable.

### 5.2 Checks against the paper (fail loudly if wrong)
- **96 demonstrations**, planar end-effector trajectories (x, y) in metres.
- All 96 start at **exactly the same point ≈ (0.52492847, −0.27973673) m** (App. B.4).
- Route labels (§5.4) give 24 possible routes; App. B.4: 22 routes have 4 demos, one has 3, one has 5.

### 5.3 Preprocessing (App. B.1)
- Normalize coordinates as `(xy − (0.5, −0.28)) / 0.4`; keep metres as well for metrics.
- **First observation** of each demo: interpolate recorded frames 4–8 to 24 points.
- **Futures:** split each demo's path after that into consecutive **8 cm arc-length segments**, each resampled to
  **24 points** (uniform in arc length); keep the shorter final segment (resampled to 24 points; mark it with its
  true length), no padding. The observation for a segment is the preceding segment. Expect **1,030 segments** from
  the 96 demos — report the count you get.
- **Folds:** three route-stratified folds, each with **64 train / 32 eval demos**, split by demonstration (never by
  segment). Route labels are used only to build the folds and for evaluation.

### 5.4 Geometry needed for routes and metrics
Read from the D3IL Avoiding environment definition in the repo (MuJoCo XML / env source; no need to install MuJoCo
or run the simulator) and hardcode with a source comment:
- obstacle centres and radii; the three obstacle rows (2, 3 and 4 gaps → 2 × 3 × 4 = 24 routes);
- goal line **y = 0.35 m**; reach envelope **≈ 1.0913 m** (tolerance 1e-6) and the base point it is measured from;
- obstacle inflation of **1 cm** for the manipulator tip (App. B.4).

Provide `route_label(path_xy) -> int | None`: route = the gap index at the **first upward crossing of each
obstacle-row centre line**, combined into 0..23 (document the encoding); `None` if a row is never crossed.

### 5.5 Rendering (new)
- `render_segment(points: Tensor (..., 24, 2)) -> Tensor (..., 3, 64, 64)`, input in **normalized coordinates**.
- **Split the 24 points into 3 frames**: points 1–8 → frame 0, 9–16 → frame 1, 17–24 → frame 2. Each frame is an
  anti-aliased polyline through its points. Splitting keeps the direction of travel, which a single polyline image
  would lose.
- **Fixed workspace window** (no agent-centred crop: obstacle positions matter): cover all 96 demos plus a margin;
  compute from the data and hardcode. At 64×64 that is ~1.25 cm/pixel, so an 8 cm segment spans ~6–7 pixels.
- Line drawing: distance-to-segment Gaussian falloff (σ ≈ 0.7–1.0 px), max-combined over the polyline's segments,
  peak 1. Must be exact for degenerate input: the **stationary start** (the start point repeated 24 times) renders
  as three identical dots.
- Fully batched in torch and fast: target < 5 ms for a batch of 384 segments on a T4 (report CPU and, if you can,
  GPU timings).

### 5.6 Data returned
Per item (one observation–future segment pair):

| Key | Shape | Content |
|---|---|---|
| `obs_pts` | (24, 2) | observation segment, normalized coordinates |
| `fut_pts` | (24, 2) | following segment, normalized coordinates |
| `fut_len` | () | true arc length of the future segment in metres (8 cm except final segments) |
| `demo_id`, `seg_idx` | () | provenance |
| `route` | () | the demo's route label (evaluation/folds only; never a model input) |

Constructor: `D3ILAvoiding(split, fold: int in {0, 1, 2}, root="data", seed=0)`. Also expose `start_point()` and
the geometry constants from §5.4 at module level, and a `demos(fold, split)` accessor returning full paths (metres)
for the reference panel.

### 5.7 Config
`conf/dataset/d3il.yaml`: `name: d3il`, `fold: 0` (add `fold` to `DatasetCfg`).

## 6. Tests (pytest, `tests/test_hep18.py`, `tests/test_d3il.py`)

Write these first; they are the acceptance criteria.

**HEP18**
- determinism: same `(split, seed)` → identical states; different seeds differ;
- sizes: 256/64/128 observations, 18 futures each, splits disjoint;
- kinematics: for every schedule, positions/speeds at 3, 6, 9 s equal the closed-form integration (atol 1e-9);
  yielding speeds are exactly (0.4, 0.1, 0) v0; all 18 future ids present per observation;
- no collisions at 0.05 s resolution in any stored future (re-run the checker);
- every yielding car stops before the intersection area (expected s ∈ [−23, −16]); every priority car has cleared
  it by t = 9 s (expected s(9) ≥ 18 m);
- renderer: output shape/dtype/range; **blob centroid tracks s**: render s on a fine grid, compute the intensity
  centroid, |centroid − mapped s| < 0.05 px everywhere in the window; shifting s by 0.5 m changes the image.

**D3IL** (skip with a clear message if `data/d3il/avoiding.npz` is missing, so CI still passes)
- 96 demos, common start point (atol 1e-6), 1,030 segments (or the count you report), 24 points each;
- arc length of every non-final segment = 8 cm (rtol 1e-3); folds are 64/32, demo-disjoint, route-stratified;
- route labels: 24 distinct routes over the 96 demos with the 22×4 + 3 + 5 distribution;
- renderer: stationary start → three identical frames, each a single dot whose centroid is the start point;
  reversing a segment's point order changes the clip (direction preserved); batched == per-sample.

**Both:** importing the modules does no network access; `scripts/fetch_d3il.py` is the only downloader.

## 7. Deliverables

1. `src/q2jepa/data/hep18.py`, `src/q2jepa/data/d3il.py` (+ shared helpers in `src/q2jepa/data/render.py` if
   useful), `conf/dataset/{hep18,d3il}.yaml`, `DatasetCfg` keys, `scripts/fetch_d3il.py`, the tests above.
2. A short report (PR description is fine) with: resampled-observation count (HEP18), segment count and licence
   (D3IL), the hardcoded geometry with sources, the render windows, σ choices, mean rendered-frame L2 norms
   (for `input_scale`), renderer timings, and a PNG grid of example clips for each task saved by a small script
   under `scripts/` (not committed images).
3. `uv run pytest` and `uv run ruff check .` pass. Work on a branch; open a PR against `phase2-vit-twin`.

## 8. Open questions (answer in the PR or ask us)

1. HEP18: confirm the `d0`-at-t = 0 reading (§4.3) once the HighwayEnv geometry is in; flag if yielding cars would
   stop inside the intersection.
2. HEP18: HighwayEnv's lane offsets/driving side and vehicle size (§4.2) — the values chosen and their source.
3. D3IL: data licence, file format, and whether camera images exist (§5.1).
4. D3IL: does the segmentation reproduce 1,030 segments? If not, what differs.
5. Both: σ and render window, finalized together with our first CNN readout-only diagnostic.
