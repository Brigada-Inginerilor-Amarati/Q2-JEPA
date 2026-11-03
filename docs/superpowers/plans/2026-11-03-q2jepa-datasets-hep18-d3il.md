# HEP18 and D3IL Rendered Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `data/hep18.py` and `data/d3il.py` so every encoder can consume both benchmarks as 64×64 image clips, while readouts and metrics get states in the paper's units.

**Architecture:** Each module holds pure state construction, a `Dataset` of state-only dict items, and one batched torch renderer that the caller runs on its own device. HEP18 is regenerated from a seed. D3IL is read from a `.npz` that a standalone script produces from the official download. No training code changes.

**Tech Stack:** Python 3.12, numpy, torch, Hydra config, pytest; `gdown` only inside `scripts/fetch_d3il.py`.

**Spec:** [docs/superpowers/specs/2026-11-03-q2jepa-datasets-hep18-d3il.md](../specs/2026-11-03-q2jepa-datasets-hep18-d3il.md), as amended in its §9.

## Global Constraints

- Renderers output exactly `(…, 3, 64, 64)` float32 in [0, 1] and do not normalize; there is no static background.
- Everything is a pure function of `(split, seed)` (and fold for D3IL), with `np.random.default_rng`; never global RNG, never time-based seeds. `seed` is a data seed, default 0.
- Renderers are batched torch functions with no Python loop over samples, and they create tensors on the input's device.
- Sub-pixel information must survive: Gaussian evaluated analytically per pixel, never rounded.
- Items hold states only, no image keys. Readout and metric fields stay in the paper's units; never decode positions from images.
- Downloaded data is untrusted: its own directory, never `pickle.load` or `torch.load` it in the main process, and everything else loads only the `.npz`.
- Importing `data/hep18.py` or `data/d3il.py` does no network access; `scripts/fetch_d3il.py` is the only downloader.
- `highway-env` is never a dependency; its constants are hardcoded with a version and source-line comment.
- No changes to `train.py`, readouts, metrics or models: those belong to the training side.
- Definition of done (`AGENTS.md` §6): `uv run pytest` and `uv run ruff check .` pass, and the `spec-rules-reviewer` runs before the PR. Line length is 120.

## Review Focus

- A NaN or infinite predicted segment reaches `render_segment` from a broken model: raise `ValueError`, because a NaN image would silently poison later calls. (Task 3)
- A demo has repeated frames or a stationary tail: arc-length resampling stays finite and never emits a zero-length final segment. (Task 6)
- A demo's path is an exact multiple of 8 cm: no empty extra segment appears. (Task 6)
- `D3ILAvoiding` is constructed with no `.npz`: `FileNotFoundError` naming `scripts/fetch_d3il.py`, not a confusing numpy error. (Task 6)
- A hostile pickle sits in the download: `safe_load` refuses it without running it. (Task 4)

---

## Before Task 1

- [ ] Create the branch and commit the approved spec and this plan:

```bash
git switch -c phase2-datasets
git add docs/superpowers/specs/2026-11-03-q2jepa-datasets-hep18-d3il.md docs/superpowers/plans/2026-11-03-q2jepa-datasets-hep18-d3il.md
git commit -m "Amend dataset spec after review; add implementation plan"
```

## File Structure

| File | Responsibility |
|---|---|
| `src/q2jepa/data/hep18.py` | Geometry constants, kinematics, generation, `HEP18`, `render_agent_clip` |
| `src/q2jepa/data/d3il.py` | Geometry constants, `render_segment`, `route_label`, segmentation, folds, `D3ILAvoiding` |
| `scripts/fetch_d3il.py` | Download, SHA check, extraction, safe conversion to `.npz` |
| `scripts/render_examples.py` | PNG grids, mean frame norms, renderer timings |
| `conf/dataset/{hep18,d3il}.yaml`, `src/q2jepa/config.py` | Dataset presets and the `fold` key |
| `tests/test_hep18.py`, `tests/test_d3il.py`, `tests/test_fetch_d3il.py` | Acceptance tests |

No `data/render.py`: the two renderers share only a one-line Gaussian.

---

### Task 1: HEP18 states and dataset

**Files:**
- Create: `src/q2jepa/data/hep18.py`, `conf/dataset/hep18.yaml`, `tests/test_hep18.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces, all float64 numpy unless noted:
  - `CARS = ("N", "W", "S", "E")`; `SCHEDULE_A, SCHEDULE_B, SCHEDULE_C, SCHEDULE_D = 0, 1, 2, 3`
  - `LANE_WIDTH`, `VEHICLE_LENGTH`, `VEHICLE_WIDTH`, `INTERSECTION_HALF`
  - `schedule_acc(schedule: int, v0: ndarray) -> ndarray (..., 3)`
  - `car_state(d0, v0, acc, t) -> tuple[ndarray, ndarray]`: exact `(s, v)` at times `t`, for any `t` in [−2, 9]
  - `future_index(axis: int, a_first: int, a_second: int) -> int`
  - `future_schedules(j: int) -> ndarray (4,)`: each car's schedule in joint future `j`
  - `to_xy(s, v, car: int) -> ndarray (..., 6)`: `(x/50, y/50, vx/10, vy/10, cos φ, sin φ)`
  - `any_overlap(xy: ndarray (..., 4, 2)) -> bool`
  - `collides(d0: ndarray (4,), v0: ndarray (4,)) -> bool`
  - `generate(n: int, seed: int) -> tuple[ndarray (n, 4), ndarray (n, 4), int]`: `(d0, v0, resampled)`
  - `HEP18(split, root="data", seed=0)`, a `Dataset` with float64 array attributes `obs_state`, `fut_state`, plus `resampled`; items are float32 tensors with the keys in spec §4.6

- [ ] **Step 1: Read the HighwayEnv geometry once, outside the project**

Run from the scratchpad, not the repo:

```bash
uv run --no-project --with highway-env python -c "import highway_env, inspect, highway_env.envs.intersection_env as m; from highway_env.vehicle.kinematics import Vehicle; from highway_env.road.lane import AbstractLane as L; print(highway_env.__version__, L.DEFAULT_WIDTH, Vehicle.LENGTH, Vehicle.WIDTH); print(inspect.getsource(m.IntersectionEnv._make_road))"
```

Expected from memory, to be confirmed: width 4, vehicle 5 × 2, `outer_distance = 11`, right-hand traffic. Record the version, source lines, each approach's heading and lane offset. Answer spec §8 Q1 and Q2: if `INTERSECTION_HALF` isn't below 16 m minus half a vehicle, stop and report, because yielding cars would stop inside the intersection.

- [ ] **Step 2: Write the failing tests in `tests/test_hep18.py`**

```python
@pytest.mark.parametrize(("schedule", "s_rel", "v_rel"), [
    (SCHEDULE_A, [3 * V - 2.25, 6 * V - 6.75, 9 * V - 9], [V - 1.5, V - 1.5, V]),
    (SCHEDULE_B, [3 * V, 6 * V, 9 * V], [V, V, V]),
    (SCHEDULE_C, [3 * V + 2.25, 6 * V + 6.75, 9 * V + 11.25], [V + 1.5, V + 1.5, V + 1.5]),
    (SCHEDULE_D, [2.1 * V, 2.85 * V, 3 * V], [0.4 * V, 0.1 * V, 0.0]),
])
def test_kinematics_closed_form(schedule, s_rel, v_rel):  # V = 8.5, D0 = 45.0
    s, v = car_state(D0, V, schedule_acc(schedule, np.array(V)), np.array([3.0, 6.0, 9.0]))
    np.testing.assert_allclose(s, -D0 + np.array(s_rel), atol=1e-9)
    np.testing.assert_allclose(v, v_rel, atol=1e-9)
```

Also write these tests:
- `test_observation_is_constant_speed`: `car_state` at −2, −1, 0 gives `−d0 + v0·t` and speed `v0`.
- `test_mid_stage_is_exact`: schedule C at t = 4.5 gives `s(3) + 1.5·(v0 + 1.5)`.
- `test_future_index_is_a_bijection`: `future_index` over axis × 3 × 3 covers 0..17 exactly once, with `j = 9·axis + 3·a_first + a_second`.
- `test_future_schedules`: `future_schedules(future_index(0, 2, 1))` is `[C, D, B, D]`; `future_schedules(future_index(1, 0, 2))` is `[D, A, D, C]`.
- `test_any_overlap`: N and W cars both at their lane position with s = 0 overlap; four cars at s = −40 do not. Cars are axis-aligned, so compare each pair's `|dx|` and `|dy|` with the sums of their half-extents.
- `test_generate_resamples_on_collision`: monkeypatch `collides` to be True on its first call; `resampled == 1`, and the result is deterministic.
- `test_no_collisions`: `resampled == 0` for seed 0, and `collides` is False for every stored observation.
- `test_determinism_and_seeds`: same `(split, seed)` gives `array_equal` `fut_state`; a different seed differs.
- `test_sizes_and_disjoint_splits`: lengths 256/64/128; item shapes as in spec §4.6; the three splits' `obs_state` arrays concatenated equal one array generated from `generate(448, seed)`, in order.
- `test_sampling_ranges`: `d0 = −obs_state[..., 2, 0]` in [43, 47]; `v0` in [8, 9].
- `test_yielding_and_priority_end_states`: every yielding car has s(9) in [−23, −16] and every priority car s(9) ≥ 16; both are clear of `INTERSECTION_HALF` by half a vehicle.
- `test_future_ids`: `future_id` is `0..17`, and `fut_state` at index j matches `future_schedules(j)`.
- `test_stored_xy_projects_to_route`: for each car, `to_xy` scaled back and projected on its heading gives s, its speed gives v, the lateral offset is the lane's, and `cos²φ + sin²φ = 1`.
- `test_item_is_float32_and_has_no_images`: all float tensors are float32, and the keys are exactly the §4.6 set.
- `test_unknown_split_raises`: `HEP18("val")` raises `ValueError`.

In `tests/test_config.py`, add `test_hep18_preset`: `_cfg(["dataset=hep18"]).dataset` has `name == "hep18"` and sizes 256/64/128.

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_hep18.py -v`
Expected: FAIL at import, `No module named 'q2jepa.data.hep18'`.

- [ ] **Step 4: Implement `src/q2jepa/data/hep18.py` and `conf/dataset/hep18.yaml`**

Pinned values:
- Car order N, W, S, E; first/second priority cars are N/S on axis 0 and W/E on axis 1.
- `generate` uses one `default_rng(seed)`, draws one observation at a time, and redraws it while `collides`. Draw order: `v0 ~ U[8, 9]`, then `d0 ~ U[43, 47]`.
- `HEP18` slices `generate(448, seed)` as 0–255 train, 256–319 dev, 320–447 test.
- `collides` steps the 18 futures on `np.linspace(-2, 9, 221)`.
- The module docstring records the `d0`-at-t = 0 reading, the priority order, and the constants' source.

The yaml sets `name: hep18`, `n_train: 256`, `n_dev: 64`, `n_test: 128`.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_hep18.py tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/q2jepa/data/hep18.py conf/dataset/hep18.yaml tests/test_hep18.py tests/test_config.py
git commit -m "Add HEP18 states and dataset"
```

### Task 2: HEP18 renderer

**Files:**
- Modify: `src/q2jepa/data/hep18.py`, `tests/test_hep18.py`

**Interfaces:**
- Produces: `S_WINDOW = (-75.0, 60.0)`, `SIGMA = 1.25`, `COLUMN = 31.5`, `s_to_row(s: Tensor) -> Tensor`, `render_agent_clip(s: Tensor (..., 3)) -> Tensor (..., 3, 64, 64)`

- [ ] **Step 1: Write the failing tests**

```python
def test_blob_centroid_tracks_s():
    s = torch.linspace(-65.0, 49.25, 500)
    clip = render_agent_clip(s[:, None].expand(-1, 3))[:, 0]
    rows, cols = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    total = clip.sum((-2, -1))
    assert ((clip * rows).sum((-2, -1)) / total - s_to_row(s)).abs().max() < 0.05
    assert ((clip * cols).sum((-2, -1)) / total - COLUMN).abs().max() < 0.05


def test_car_moves_up_the_image():
    assert s_to_row(torch.tensor(10.0)) < s_to_row(torch.tensor(0.0))
```

Also write these tests:
- `test_render_shape_dtype_range`: a float64 input of shape `(4, 3)` gives float32 `(4, 3, 64, 64)`, within [0, 1], and `(18, 4, 3)` gives `(18, 4, 3, 64, 64)`.
- `test_half_metre_shift_changes_image`.
- `test_frames_follow_their_own_position`: frame f changes only when `s[..., f]` changes.
- `test_batched_equals_per_sample`.
- `test_renderer_stays_on_input_device`: skipped unless CUDA is available, because the Mac can't exercise it.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_hep18.py -k "blob or moves_up or render or shift or frames_follow or batched" -v`
Expected: FAIL at import, `cannot import name 'render_agent_clip'`.

- [ ] **Step 3: Implement `render_agent_clip`**

`r = (s_max − s) / (s_max − s_min) · 63`, row 0 at the top; `exp(−((row − r)² + (col − COLUMN)²) / (2·SIGMA²))` on pixel centres `0..63`, peak amplitude 1. Docstring gives the world-to-pixel mapping and says σ and the column are provisional (spec §8).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_hep18.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/q2jepa/data/hep18.py tests/test_hep18.py
git commit -m "Add HEP18 per-car renderer"
```

### Task 3: D3IL renderer (no data needed)

**Files:**
- Create: `src/q2jepa/data/d3il.py`, `tests/test_d3il.py`

**Interfaces:**
- Produces:
  - `NORM_CENTRE = (0.5, -0.28)`, `NORM_SCALE = 0.4`
  - `WINDOW_X = (-0.9375, 1.0625)`, `WINDOW_Y = (-0.2125, 1.7875)`, in normalized coordinates (spec §5.5 in metres)
  - `SIGMA = 0.85`, `START_POINT = (0.52492847, -0.27973673)`
  - `normalize(xy_m) -> xy`
  - `to_pixels(points: Tensor (..., 2)) -> Tensor (..., 2)`: `(col, row)` after clamping
  - `render_segment(points: Tensor (..., 24, 2)) -> Tensor (..., 3, 64, 64)`

- [ ] **Step 1: Write the failing tests**

```python
def _centroid(frame):
    rows, cols = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    return torch.stack([(frame * cols).sum(), (frame * rows).sum()]) / frame.sum()


def test_stationary_start_is_three_identical_dots():
    start = torch.tensor(normalize(np.array(START_POINT)), dtype=torch.float32)
    clip = render_segment(start.expand(24, 2))
    assert torch.equal(clip[0], clip[1]) and torch.equal(clip[1], clip[2])
    assert (_centroid(clip[0]) - to_pixels(start)).abs().max() < 0.05


def test_out_of_window_points_clamp_to_the_edge_and_never_blank():
    far = torch.tensor([[50.0, 50.0]]).expand(24, 2)
    corner = torch.tensor([[WINDOW_X[1], WINDOW_Y[1]]]).expand(24, 2)
    clip = render_segment(far)
    assert torch.equal(clip, render_segment(corner))
    assert (clip.sum((-2, -1)) > 0).all()
```

Also write these tests:
- `test_shape_dtype_range`: float64 input gives float32 `(3, 64, 64)` in [0, 1]; a batch `(5, 24, 2)` gives `(5, 3, 64, 64)`.
- `test_reversal_changes_clip`.
- `test_frames_use_their_own_points`: changing point index 8 (the 9th) leaves frame 0 and frame 2 unchanged.
- `test_batched_equals_per_sample`.
- `test_subpixel_shift_moves_centroid`: a stationary dot shifted by 0.25 px moves the centroid by 0.25 ± 0.05.
- `test_line_reaches_full_intensity_along_its_length`: a straight line along pixel centres has intensity about 1 at its midpoint.
- `test_y_points_up`: a larger y gives a smaller row.
- `test_non_finite_points_raise`: NaN or inf raises `ValueError`.
- `test_renderer_stays_on_input_device`: skipped unless CUDA is available.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_d3il.py -v`
Expected: FAIL at import, `No module named 'q2jepa.data.d3il'`.

- [ ] **Step 3: Implement `render_segment`**

Check finiteness, clamp to the window, map to pixels with `col = (x − x_min) / 2 · 63` and `row = (y_max − y) / 2 · 63`, split into 3 frames of 8 points. For each frame's 7 sub-segments use the distance from each pixel centre to the segment, with `t = clamp(dot(q − a, b − a) / |b − a|².clamp_min(1e-12), 0, 1)`. Take the Gaussian `exp(−d² / (2·SIGMA²))` and `amax` over the sub-segments. The module docstring gives the world-to-pixel mapping.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_d3il.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/q2jepa/data/d3il.py tests/test_d3il.py
git commit -m "Add D3IL segment renderer"
```

### Task 4: Fetch and convert D3IL

**Files:**
- Create: `scripts/fetch_d3il.py`, `tests/test_fetch_d3il.py`

**Interfaces:**
- Produces:
  - `sha256(path: Path) -> str`
  - `safe_load(path: Path) -> object`
  - `extract_xy(obj: object) -> ndarray (T, 2)`
  - `convert(raw_dir: Path, out: Path) -> None`
  - `data/d3il/avoiding.npz` with `xy` float64 `(N, 2)` and `offsets` int64 `(97,)`; demo `i` is `xy[offsets[i]:offsets[i + 1]]`, demos sorted by filename

The script imports only stdlib and numpy at the top, because `convert` runs under `python -I`; `gdown` is imported inside the download function.

- [ ] **Step 1: Ask the human before downloading**

State the source (the Drive link in spec §5.1) and that the size is unknown until the download starts; wait for a clear yes. Separately, shallow-clone `ALRhub/d3il` to a new scratch directory, read-only and without running anything, and note the data licence if the repo states one, for the PR report.

- [ ] **Step 2: Write the download half of the script, then run it**

`uv run --with gdown python scripts/fetch_d3il.py --download`: downloads with `gdown` to `data/d3il/raw/`, prints the SHA-256, extracts only members under the Avoiding task, and refuses to overwrite an existing zip. Pin the printed value as `ZIP_SHA256`; a later mismatch raises `ValueError`.

- [ ] **Step 3: Inspect the format without loading it**

Run from a scratch directory, passing paths as arguments, with `python -I -m pickletools <file> | head -100`. Report for spec §8 Q3: file format, every class the pickle references, and whether camera images exist.

**Stop and ask** if a pickle references anything beyond plain containers and numpy array reconstruction.

Choose the position field by the paper's check: the one where all 96 demos start within 1e-6 m of `START_POINT` (from memory the candidates are `des_c_pos` and `c_pos`; confirm). If neither matches, stop and report.

- [ ] **Step 4: Write the failing tests**

```python
class _Hostile:
    def __reduce__(self):
        return (os.getcwd, ())


def test_safe_load_refuses_hostile_pickle(tmp_path):
    path = tmp_path / "x.pkl"
    path.write_bytes(pickle.dumps(_Hostile()))
    with pytest.raises(pickle.UnpicklingError):
        safe_load(path)
```

Also write these tests, using a synthetic pickle with the structure found in Step 3:
- `test_safe_load_reads_numpy_arrays`.
- `test_convert_writes_plain_arrays`: two synthetic demos in `tmp_path`; `np.load(out, allow_pickle=False)` works, and `offsets` and `xy` reproduce both demos in sorted filename order.
- `test_sha256_matches_hashlib`.

- [ ] **Step 5: Run to verify they fail**

Run: `uv run pytest tests/test_fetch_d3il.py -v`
Expected: FAIL at import, `cannot import name 'safe_load'`.

- [ ] **Step 6: Implement `safe_load`, `extract_xy` and `convert`**

`safe_load` subclasses `pickle.Unpickler` and overrides `find_class` to allow only the classes seen in Step 3, otherwise raising `UnpicklingError`. Running with no flags downloads if needed, then runs `convert` with `subprocess.run([sys.executable, "-I", __file__, "--convert", raw, out], cwd=<temp dir>, check=True)`, so it works unattended on Colab.

- [ ] **Step 7: Run to verify they pass, then produce the data**

Run: `uv run pytest tests/test_fetch_d3il.py -v`, then `uv run --with gdown python scripts/fetch_d3il.py`
Expected: PASS, and `data/d3il/avoiding.npz` exists. `data/` stays git-ignored.

- [ ] **Step 8: Commit**

```bash
git add scripts/fetch_d3il.py tests/test_fetch_d3il.py
git commit -m "Add D3IL fetch script with safe pickle conversion"
```

### Task 5: D3IL geometry and route labels

**Files:**
- Modify: `src/q2jepa/data/d3il.py`, `tests/test_d3il.py`

**Interfaces:**
- Produces: `ROW_Y: tuple[float, ...]` (row centre lines, in crossing order), `ROW_OBSTACLE_X: tuple[tuple[float, ...], ...]`, `OBSTACLE_RADIUS`, `GOAL_Y = 0.35`, `REACH = 1.0913`, `REACH_TOL = 1e-6`, `REACH_BASE`, `INFLATION = 0.01`, `start_point() -> ndarray (2,)`, `route_label(path_xy: ndarray (T, 2)) -> int | None`

- [ ] **Step 1: Read the geometry from the clone made in Task 4**

From the Avoiding environment's XML and env source: obstacle centres and radii, the base point for reach, and the row order. Hardcode them with a source comment giving the commit and paths. Rows must have 2, 3 and 4 gaps. Confirm the x window (spec §5.5) contains the obstacles and the start.

- [ ] **Step 2: Write the failing tests**

```python
def _through(gaps):
    """Path from below the first row, through the given gap of each row, to the goal."""
```

Build `x` at each row as the midpoint of the chosen gap, from `ROW_OBSTACLE_X`. Then write these tests:
- `test_route_label_is_a_bijection`: the 24 gap combinations give labels covering 0..23 exactly once, with `label = 12·g0 + 4·g1 + g2`, where gaps are numbered by increasing x.
- `test_first_upward_crossing_wins`: a path crosses row 0 upward in gap 0, dips back below, and crosses again in gap 1; its `g0` is 0.
- `test_downward_crossing_does_not_count`: a path starting above row 0 that comes down and never crosses upward returns `None`.
- `test_uncrossed_row_returns_none`: a path stopping below the last row returns `None`.
- `test_constants_match_paper`: `GOAL_Y`, `REACH`, `INFLATION`, and rows with 2, 3 and 4 gaps.

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_d3il.py -k "route or constants" -v`
Expected: FAIL at import, `cannot import name 'route_label'`.

- [ ] **Step 4: Implement `route_label`**

An upward crossing of line Y is the first `i` with `y[i] < Y <= y[i + 1]`. Interpolate x at Y, and take the gap with `np.searchsorted(ROW_OBSTACLE_X[row], x)`.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_d3il.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/q2jepa/data/d3il.py tests/test_d3il.py
git commit -m "Add D3IL geometry and route labels"
```

### Task 6: D3IL segmentation, folds and dataset

**Files:**
- Create: `conf/dataset/d3il.yaml`
- Modify: `src/q2jepa/data/d3il.py`, `src/q2jepa/config.py`, `tests/test_d3il.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `normalize`, `route_label`, `START_POINT` (Tasks 3 and 5); the `.npz` schema (Task 4).
- Produces:
  - `first_observation(path: ndarray (T, 2)) -> ndarray (24, 2)`
  - `segments(path: ndarray (T, 2)) -> tuple[ndarray (K, 24, 2), ndarray (K,)]`: points and true lengths in metres
  - `make_folds(routes: ndarray (96,), seed: int) -> ndarray (96,)`: each demo's eval fold in {0, 1, 2}
  - `D3ILAvoiding(split, fold, root="data", seed=0)`, with items keyed as in spec §5.6
  - `demos(fold: int, split: str, root="data", seed=0) -> list[ndarray]`
  - `EXPECTED_SEGMENTS = 1030`
  - `DatasetCfg.fold: int = 0`

- [ ] **Step 1: Write the failing synthetic tests (no data needed)**

```python
def test_segments_exclude_start_and_include_endpoint():
    path = np.stack([np.linspace(0, 0.08, 200), np.zeros(200)], -1)
    pts, lengths = segments(path)
    assert pts.shape == (1, 24, 2)
    np.testing.assert_allclose(pts[0, :, 0], 0.08 * np.arange(1, 25) / 24, atol=1e-9)
    np.testing.assert_allclose(lengths, [0.08], atol=1e-9)
```

Also write these tests:
- `test_exact_multiple_has_no_empty_segment`: a 16 cm straight path gives exactly 2 segments.
- `test_short_final_segment_keeps_true_length`: a 20 cm path gives lengths 8, 8 and 4 cm, each with 24 points.
- `test_repeated_frames_stay_finite`: a path with duplicated points gives finite output and matches the same path without duplicates.
- `test_stationary_path_has_no_segments`: returns shape `(0, 24, 2)`.
- `test_first_observation_interpolates_frames_4_to_8`: a path with `x = frame index` gives 24 points from 3 to 7, evenly spaced (`frames[3:8]`, 1-indexed).
- `test_make_folds_is_stratified_partition`: routes with the 22×4 + 3 + 5 distribution give 32 per fold, each route's fold counts differ by at most 1, and it is deterministic per seed.
- `test_missing_npz_raises_with_fetch_hint`: `D3ILAvoiding("train", 0, root=tmp_path)` raises `FileNotFoundError` matching `fetch_d3il`.
- `test_dev_split_raises` (no data needed, checked before loading): `split="dev"` raises `ValueError`.
- `test_fold_out_of_range_raises`: `fold=3` raises `ValueError`.

In `tests/test_config.py`, add `test_d3il_preset`: name `d3il`, `fold == 0`, and the override `dataset.fold=2` is accepted.

- [ ] **Step 2: Write the failing data tests**

Mark them with `pytest.mark.skipif(not NPZ.exists(), reason="data/d3il/avoiding.npz missing; run scripts/fetch_d3il.py")`:
- `test_96_demos_common_start`: within 1e-6 of `START_POINT`.
- `test_segment_count`: equals `EXPECTED_SEGMENTS`; all 24 points.
- `test_non_final_arc_length`: from the previous segment's last point, 8 cm with rtol 1e-3.
- `test_route_distribution`: 24 routes, sorted counts `[3] + [4] * 22 + [5]`.
- `test_folds_disjoint_and_complete`: 64/32 per fold; the eval sets across folds partition the 96.
- `test_observation_is_preceding_segment`: for `seg_idx > 0`, `obs_pts` equals the previous item's `fut_pts`; for `seg_idx == 0`, it equals the normalized `first_observation`.
- `test_every_demo_fits_the_window`: if this fails, report it rather than widening the window.
- `test_item_keys_and_shapes`: spec §5.6, float32 points, and normalized points map back to metres.

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_d3il.py tests/test_config.py -v`
Expected: FAIL at import, `cannot import name 'segments'`.

- [ ] **Step 4: Implement segmentation and folds**

- `segments` takes the path to split, and `D3ILAvoiding` passes `demo[7:]`, so frame 8 is the shared start. It drops zero-length steps, takes boundaries at multiples of 0.08 m, keeps a final remainder only if it exceeds 1e-9 m, and samples at `start + length · k/24`, `k = 1..24`, with `np.interp` per axis.
- `first_observation` uses `np.interp` over frame index on `path[3:8]`, endpoints included.
- `make_folds` concatenates `rng.permutation(demos_of_route)` over sorted routes; the demo at position `i` gets eval fold `i % 3`. This gives exactly 32 per fold, because 96 = 3 × 32, and spreads each route within 1.

- [ ] **Step 5: Implement `D3ILAvoiding`, `demos`, `start_point`, the config key and yaml**

- `"test"` is the fold's 32 eval demos; validate `split` and `fold` before touching the disk.
- Load with `allow_pickle=False`. Raise `ValueError` if §5.2 fails: not 96 demos, starts off by more than 1e-6, a `None` route, or the wrong route distribution.
- The segment count is not a constructor check, because §6 allows reporting a different count.
- `conf/dataset/d3il.yaml` sets `name: d3il` and `fold: 0`. Add `fold: int = 0` to `DatasetCfg`.

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/test_d3il.py tests/test_config.py -v`
Expected: PASS. If the count isn't 1,030, try 0-indexed frames (`path[4:9]`) and set `EXPECTED_SEGMENTS` to the count from whichever reading is closer. Report both counts for spec §8 Q4.

- [ ] **Step 7: Commit**

```bash
git add src/q2jepa/data/d3il.py src/q2jepa/config.py conf/dataset/d3il.yaml tests/test_d3il.py tests/test_config.py
git commit -m "Add D3IL segmentation, folds and dataset"
```

### Task 7: Examples, norms and handoff

**Files:**
- Create: `scripts/render_examples.py`
- Modify: `src/q2jepa/data/hep18.py`, `src/q2jepa/data/d3il.py` (docstrings), `tests/test_hep18.py`, `tests/test_d3il.py`

**Interfaces:**
- Consumes: `HEP18`, `render_agent_clip`, `D3ILAvoiding`, `render_segment`.
- Produces: `mean_frame_norm(clips: Tensor (..., H, W)) -> float`; `uv run python scripts/render_examples.py [--out outputs/render_examples]` saves one PNG grid per task with `torchvision.utils.save_image`, then prints mean norms, `input_scale = mean_norm / 64`, and timings. D3IL is skipped with a message if the `.npz` is missing.

- [ ] **Step 1: Write the failing tests**

```python
def test_mean_frame_norm_matches_analytic_blob():
    ds = HEP18("train")
    clips = render_agent_clip(torch.tensor(ds.obs_state[:8, :, :, 0], dtype=torch.float32))
    assert abs(mean_frame_norm(clips) - SIGMA * math.sqrt(math.pi)) < 0.02
```

Also write `test_import_does_no_network` in both test files: monkeypatch `socket.socket` to raise, then `importlib.reload` the module.

Also write `test_only_fetch_script_downloads`: an AST walk confirms that `data/hep18.py` and `data/d3il.py` import none of `gdown`, `urllib`, `requests`, `highway_env`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_hep18.py -k mean_frame_norm -v`
Expected: FAIL at import, `No module named 'scripts.render_examples'`.

- [ ] **Step 3: Implement `scripts/render_examples.py`**

The HEP18 norm covers all train obs and future clips. The D3IL norm covers fold 0's train `obs_pts` and `fut_pts`. Render in chunks to bound memory. Time `render_segment` on a batch of 384 on CPU, and on CUDA if available.

- [ ] **Step 4: Run the script and record the results**

Run: `uv run python scripts/render_examples.py`
Then look at both PNG grids. HEP18 should show one blob per frame moving up, and D3IL short streaks moving between frames. Put the printed norms in each module's docstring, and the zip's SHA-256 in `d3il.py`'s (spec §5.1). Commit no images.

- [ ] **Step 5: Run everything**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check .`
Expected: all pass. Separately, with `data/d3il/avoiding.npz` moved aside, `uv run pytest tests/test_d3il.py` still runs the renderer tests and skips only the data tests. Move the file back afterwards.

- [ ] **Step 6: Commit**

```bash
git add scripts/render_examples.py src/q2jepa/data/hep18.py src/q2jepa/data/d3il.py tests/test_hep18.py tests/test_d3il.py
git commit -m "Add render examples script and record frame norms"
```

- [ ] **Step 7: Run the spec review and update the status table**

Run the `spec-rules-reviewer` subagent on the branch diff and fix any findings. Update the `AGENTS.md` §2 row "HEP18 and D3IL as rendered images" to say data and renderers are done, while readouts and metrics are not built.

```bash
git add AGENTS.md
git commit -m "Update status table for HEP18 and D3IL data"
```

- [ ] **Step 8: Ask before pushing, then open the PR against `main`**

The PR description is the report in spec §7. It covers:
- resampled-observation count;
- segment count and licence;
- format and camera answers;
- hardcoded geometry with sources;
- render windows and provisional σ;
- mean norms and timings;
- how to run the script.

Add the notes for the training side (spec §7 item 4):
- `train.py` ignores `dataset.name`;
- 16 unique clips instead of 72;
- the render check they own (spec §8);
- the k/24 segment convention, for the readout owner to confirm.

If the licence is unclear, say that no D3IL results may be reported yet.
