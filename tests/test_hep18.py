import ast
import importlib
import math
import socket
from pathlib import Path

import numpy as np
import pytest
import torch

from q2jepa.data import d3il, hep18
from q2jepa.data.hep18 import (
    COLUMN,
    HEP18,
    INTERSECTION_HALF,
    SCHEDULE_A,
    SCHEDULE_B,
    SCHEDULE_C,
    SCHEDULE_D,
    SIGMA,
    VEHICLE_LENGTH,
    any_overlap,
    car_state,
    collides,
    future_index,
    future_schedules,
    generate,
    render_agent_clip,
    s_to_row,
    schedule_acc,
    to_xy,
)
from scripts.render_examples import mean_frame_norm

D0, V = 45.0, 8.5
ITEM_KEYS = {"obs_state", "fut_state", "obs_xy", "fut_xy", "future_id", "obs_idx"}


@pytest.fixture(scope="module")
def train():
    return HEP18("train")


@pytest.mark.parametrize(
    ("schedule", "s_rel", "v_rel"),
    [
        (SCHEDULE_A, [3 * V - 2.25, 6 * V - 6.75, 9 * V - 9], [V - 1.5, V - 1.5, V]),
        (SCHEDULE_B, [3 * V, 6 * V, 9 * V], [V, V, V]),
        (SCHEDULE_C, [3 * V + 2.25, 6 * V + 6.75, 9 * V + 11.25], [V + 1.5, V + 1.5, V + 1.5]),
        (SCHEDULE_D, [2.1 * V, 2.85 * V, 3 * V], [0.4 * V, 0.1 * V, 0.0]),
    ],
)
def test_kinematics_closed_form(schedule, s_rel, v_rel):
    s, v = car_state(D0, V, schedule_acc(schedule, np.array(V)), np.array([3.0, 6.0, 9.0]))
    np.testing.assert_allclose(s, -D0 + np.array(s_rel), atol=1e-9)
    np.testing.assert_allclose(v, v_rel, atol=1e-9)


def test_observation_is_constant_speed():
    t = np.array([-2.0, -1.0, 0.0])
    s, v = car_state(D0, V, schedule_acc(SCHEDULE_C, np.array(V)), t)
    np.testing.assert_allclose(s, -D0 + V * t, atol=1e-9)
    np.testing.assert_allclose(v, V, atol=1e-9)


def test_mid_stage_is_exact():
    acc = schedule_acc(SCHEDULE_C, np.array(V))
    s3, _ = car_state(D0, V, acc, np.array([3.0]))
    s, v = car_state(D0, V, acc, np.array([4.5]))
    np.testing.assert_allclose(s, s3 + 1.5 * (V + 1.5), atol=1e-9)
    np.testing.assert_allclose(v, V + 1.5, atol=1e-9)


def test_future_index_is_a_bijection():
    ids = [future_index(ax, a, b) for ax in range(2) for a in range(3) for b in range(3)]
    assert sorted(ids) == list(range(18))
    assert future_index(1, 2, 1) == 9 + 6 + 1


def test_future_schedules():
    a, b, c, d = SCHEDULE_A, SCHEDULE_B, SCHEDULE_C, SCHEDULE_D
    assert future_schedules(future_index(0, 2, 1)).tolist() == [c, d, b, d]
    assert future_schedules(future_index(1, 0, 2)).tolist() == [d, a, d, c]


def _positions(s):
    return np.stack([to_xy(np.array(s[c]), np.array(8.0), c)[:2] * 50 for c in range(4)])


def test_any_overlap():
    assert any_overlap(_positions([0.0, 0.0, -40.0, -40.0]))  # N and W meet in the middle
    assert not any_overlap(_positions([-40.0] * 4))
    assert not any_overlap(_positions([0.0, -40.0, 0.0, -40.0]))  # opposing lanes pass side by side


def test_generate_resamples_on_collision(monkeypatch):
    real, calls = collides, []

    def first_collides(d0, v0):
        calls.append(1)
        return len(calls) == 1 or real(d0, v0)

    monkeypatch.setattr(hep18, "collides", first_collides)
    d0, v0, resampled = generate(3, 0)
    assert resampled == 1 and d0.shape == v0.shape == (3, 4)
    calls.clear()
    np.testing.assert_array_equal(d0, generate(3, 0)[0])


def test_no_collisions(train):
    assert train.resampled == 0
    assert not any(collides(d, v) for d, v in zip(train.d0[:32], train.v0[:32], strict=True))


def test_determinism_and_seeds(train):
    np.testing.assert_array_equal(train.fut_state, HEP18("train").fut_state)
    assert not np.array_equal(train.fut_state, HEP18("train", seed=1).fut_state)


def test_sizes_and_disjoint_splits(train):
    splits = [train, HEP18("dev"), HEP18("test")]
    assert [len(d) for d in splits] == [256, 64, 128]
    d0, _, _ = generate(448, 0)
    np.testing.assert_array_equal(np.concatenate([d.d0 for d in splits]), d0)
    item = train[0]
    assert item["obs_state"].shape == (4, 3, 2) and item["fut_state"].shape == (18, 4, 3, 2)
    assert item["obs_xy"].shape == (4, 3, 6) and item["fut_xy"].shape == (18, 4, 3, 6)
    assert item["future_id"].shape == (18,) and item["obs_idx"].shape == ()


def test_sampling_ranges(train):
    d0 = -train.obs_state[:, :, 2, 0]
    v0 = train.obs_state[:, :, 0, 1]
    assert d0.min() >= 43 and d0.max() <= 47
    assert v0.min() >= 8 and v0.max() <= 9


def test_yielding_and_priority_end_states(train):
    s9 = train.fut_state[:, :, :, 2, 0]
    yielding = np.stack([future_schedules(j) == SCHEDULE_D for j in range(18)])
    assert s9[:, yielding].min() >= -23 and s9[:, yielding].max() <= -16
    assert s9[:, ~yielding].min() >= 16
    assert s9[:, yielding].max() + VEHICLE_LENGTH / 2 < -INTERSECTION_HALF
    assert s9[:, ~yielding].min() - VEHICLE_LENGTH / 2 > INTERSECTION_HALF


def test_yielding_speeds_are_exact_fractions(train):
    speeds = train.fut_state[0, :, :, :, 1]
    v0 = train.v0[0]
    for j in range(18):
        for car in np.flatnonzero(future_schedules(j) == SCHEDULE_D):
            np.testing.assert_allclose(speeds[j, car], [0.4 * v0[car], 0.1 * v0[car], 0.0], atol=1e-9)


def test_future_ids(train):
    assert train[0]["future_id"].tolist() == list(range(18))
    acc = schedule_acc(future_schedules(future_index(1, 2, 0)), train.v0[0])
    s, _ = car_state(train.d0[0], train.v0[0], acc, np.array([3.0, 6.0, 9.0]))
    np.testing.assert_allclose(train.fut_state[0, future_index(1, 2, 0), :, :, 0], s, atol=1e-9)


@pytest.mark.parametrize(
    ("car", "heading", "lane_point"),
    [(0, (0, 1), (-2, 0)), (1, (1, 0), (0, 2)), (2, (0, -1), (2, 0)), (3, (-1, 0), (0, -2))],
)
def test_stored_xy_projects_to_route(car, heading, lane_point):
    s, v = np.array([-30.0, 5.0]), np.array([8.0, 9.0])
    state = to_xy(s, v, car)
    heading = np.array(heading, dtype=float)
    np.testing.assert_allclose(
        state[..., :2] * 50 - s[:, None] * heading, np.broadcast_to(lane_point, (2, 2)), atol=1e-9
    )
    np.testing.assert_allclose(state[..., 2:4] * 10, v[:, None] * heading, atol=1e-9)
    np.testing.assert_allclose(state[..., 4:6], np.broadcast_to(heading, (2, 2)), atol=1e-12)


def test_item_is_float32_and_has_no_images(train):
    item = train[0]
    assert set(item) == ITEM_KEYS
    for key in ("obs_state", "fut_state", "obs_xy", "fut_xy"):
        assert item[key].dtype == torch.float32


def test_unknown_split_raises():
    with pytest.raises(ValueError, match="split"):
        HEP18("val")


def test_blob_centroid_tracks_s():
    s = torch.linspace(-65.0, 49.25, 500)
    clip = render_agent_clip(s[:, None].expand(-1, 3))[:, 0]
    rows, cols = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    total = clip.sum((-2, -1))
    assert ((clip * rows).sum((-2, -1)) / total - s_to_row(s)).abs().max() < 0.05
    assert ((clip * cols).sum((-2, -1)) / total - COLUMN).abs().max() < 0.05


def test_car_moves_up_the_image():
    assert s_to_row(torch.tensor(10.0)) < s_to_row(torch.tensor(0.0))


def test_render_shape_dtype_range():
    out = render_agent_clip(torch.zeros(4, 3, dtype=torch.float64))
    assert out.shape == (4, 3, 64, 64) and out.dtype == torch.float32
    assert out.min() >= 0 and out.max() <= 1
    assert render_agent_clip(torch.zeros(18, 4, 3)).shape == (18, 4, 3, 64, 64)


def test_half_metre_shift_changes_image():
    s = torch.tensor([-10.0, 0.0, 10.0])
    assert not torch.equal(render_agent_clip(s), render_agent_clip(s + 0.5))


def test_frames_follow_their_own_position():
    s = torch.tensor([-10.0, 0.0, 10.0])
    moved = s.clone()
    moved[1] += 5.0
    a, b = render_agent_clip(s), render_agent_clip(moved)
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2]) and not torch.equal(a[1], b[1])


def test_batched_equals_per_sample():
    s = torch.tensor([[-60.0, -50.0, -45.0], [-20.0, 0.0, 40.0]])
    assert torch.equal(render_agent_clip(s), torch.stack([render_agent_clip(x) for x in s]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_renderer_stays_on_input_device():
    assert render_agent_clip(torch.zeros(3, device="cuda")).device.type == "cuda"


def test_mean_frame_norm_matches_analytic_blob(train):
    clips = render_agent_clip(torch.tensor(train.obs_state[:8, :, :, 0], dtype=torch.float32))
    assert abs(mean_frame_norm(clips) - SIGMA * math.sqrt(math.pi)) < 0.02


@pytest.mark.parametrize("module", [hep18, d3il])
def test_import_does_no_network(module, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access during import")

    monkeypatch.setattr(socket, "socket", refuse)
    importlib.reload(module)


@pytest.mark.parametrize("module", [hep18, d3il])
def test_only_fetch_script_downloads(module):
    tree = ast.parse(Path(module.__file__).read_text())
    roots = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    roots |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not roots & {"gdown", "urllib", "requests", "highway_env"}
