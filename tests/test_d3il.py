from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from q2jepa.data.d3il import (
    EXPECTED_SEGMENTS,
    GOAL_Y,
    INFLATION,
    REACH,
    ROW_OBSTACLE_X,
    ROW_RADIUS,
    ROW_Y,
    START_POINT,
    WINDOW_X,
    WINDOW_Y,
    D3ILAvoiding,
    _load,
    demos,
    first_observation,
    make_folds,
    normalize,
    render_segment,
    route_label,
    segments,
    start_point,
    to_goal,
    to_pixels,
)

ROOT = Path(__file__).resolve().parents[1] / "data"
needs_data = pytest.mark.skipif(
    not (ROOT / "d3il/avoiding.npz").exists(), reason="data/d3il/avoiding.npz missing; run scripts/fetch_d3il.py"
)


def _centroid(frame):
    rows, cols = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    return torch.stack([(frame * cols).sum(), (frame * rows).sum()]) / frame.sum()


def _from_pixels(col, row):
    col, row = torch.as_tensor(col, dtype=torch.float32), torch.as_tensor(row, dtype=torch.float32)
    return torch.stack([WINDOW_X[0] + col / 63 * 2, WINDOW_Y[1] - row / 63 * 2], -1)


def _segment():
    return _from_pixels(torch.linspace(10.0, 30.0, 24), torch.linspace(40.0, 25.0, 24))


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


def test_shape_dtype_range():
    clip = render_segment(_segment().double())
    assert clip.shape == (3, 64, 64) and clip.dtype == torch.float32
    assert clip.min() >= 0 and clip.max() <= 1
    assert render_segment(_segment().expand(5, 24, 2)).shape == (5, 3, 64, 64)


def test_reversal_changes_clip():
    seg = _segment()
    assert not torch.equal(render_segment(seg), render_segment(seg.flip(0)))


def test_frames_use_their_own_points():
    seg = _segment()
    moved = seg.clone()
    moved[8] += 0.2
    a, b = render_segment(seg), render_segment(moved)
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2]) and not torch.equal(a[1], b[1])


def test_batched_equals_per_sample():
    batch = torch.stack([_segment(), _segment().flip(0), _from_pixels(torch.full((24,), 20.0), torch.full((24,), 9.0))])
    assert torch.equal(render_segment(batch), torch.stack([render_segment(x) for x in batch]))


def test_subpixel_shift_moves_centroid():
    a = render_segment(_from_pixels(30.0, 30.0).expand(24, 2))[0]
    b = render_segment(_from_pixels(30.25, 30.0).expand(24, 2))[0]
    assert abs((_centroid(b) - _centroid(a))[0] - 0.25) < 0.05


def test_line_reaches_full_intensity_along_its_length():
    seg = _from_pixels(torch.linspace(20.0, 40.0, 24), torch.full((24,), 30.0))
    assert render_segment(seg)[0, 30, 23] > 0.999


def test_y_points_up():
    low = torch.tensor([0.0, 0.5])
    high = torch.tensor([0.0, 1.0])
    assert to_pixels(high)[1] < to_pixels(low)[1]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_points_raise(bad):
    seg = _segment()
    seg[3, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        render_segment(seg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_renderer_stays_on_input_device():
    assert render_segment(_segment().to("cuda")).device.type == "cuda"


def _gap_x(row, gap):
    edges = (0.2, *ROW_OBSTACLE_X[row], 0.8)
    return (edges[gap] + edges[gap + 1]) / 2


def _through(gaps):
    """Path from below the first row, through the given gap of each row, to the goal."""
    points = [(0.5, -0.28)]
    for row, gap in enumerate(gaps):
        x = _gap_x(row, gap)
        points += [(x, ROW_Y[row] - 0.01), (x, ROW_Y[row] + 0.01)]
    return np.array([*points, (points[-1][0], GOAL_Y)])


def test_route_label_is_a_bijection():
    gaps = [(a, b, c) for a in range(2) for b in range(3) for c in range(4)]
    labels = [route_label(_through(g)) for g in gaps]
    assert labels == [12 * a + 4 * b + c for a, b, c in gaps]
    assert sorted(labels) == list(range(24))


def test_first_upward_crossing_wins():
    y0 = ROW_Y[0]
    dip = [(_gap_x(0, 0), y0 - 0.01), (_gap_x(0, 0), y0 + 0.01), (_gap_x(0, 1), y0 - 0.01), (_gap_x(0, 1), y0 + 0.01)]
    rest = _through((1, 2, 3))[3:]
    assert route_label(np.array([(0.5, -0.28), *dip, *rest])) == 0 * 12 + 2 * 4 + 3


def test_downward_crossing_does_not_count():
    assert route_label(np.array([(0.45, ROW_Y[0] + 0.05), (0.45, -0.28)])) is None


def test_uncrossed_row_returns_none():
    assert route_label(_through((0, 1, 2))[:5]) is None


def test_constants_match_paper():
    assert (GOAL_Y, REACH, INFLATION) == (0.35, 1.0913, 0.01)
    assert [len(x) + 1 for x in ROW_OBSTACLE_X] == [2, 3, 4]
    assert len(ROW_RADIUS) == len(ROW_Y) == 3
    np.testing.assert_array_equal(start_point(), START_POINT)


def _straight(length, n=200):
    return np.stack([np.linspace(0, length, n), np.zeros(n)], -1)


def test_segments_exclude_start_and_include_endpoint():
    pts, lengths = segments(_straight(0.08))
    assert pts.shape == (1, 24, 2)
    np.testing.assert_allclose(pts[0, :, 0], 0.08 * np.arange(1, 25) / 24, atol=1e-9)
    np.testing.assert_allclose(lengths, [0.08], atol=1e-9)


def test_exact_multiple_has_no_empty_segment():
    pts, lengths = segments(_straight(0.16))
    assert pts.shape == (2, 24, 2)
    np.testing.assert_allclose(lengths, [0.08, 0.08], atol=1e-9)


def test_short_final_segment_keeps_true_length():
    pts, lengths = segments(_straight(0.20))
    assert pts.shape == (3, 24, 2)
    np.testing.assert_allclose(lengths, [0.08, 0.08, 0.04], atol=1e-9)
    np.testing.assert_allclose(pts[2, -1], [0.20, 0.0], atol=1e-9)


def test_repeated_frames_stay_finite():
    path = _straight(0.20, 50)
    repeated = np.repeat(path, 3, axis=0)
    pts, lengths = segments(repeated)
    assert np.isfinite(pts).all()
    np.testing.assert_allclose(pts, segments(path)[0], atol=1e-9)
    np.testing.assert_allclose(lengths, segments(path)[1], atol=1e-9)


def test_stationary_path_has_no_segments():
    pts, lengths = segments(np.full((10, 2), 0.3))
    assert pts.shape == (0, 24, 2) and lengths.shape == (0,)


def test_first_observation_interpolates_frames_4_to_8():
    path = np.stack([np.arange(20.0), np.zeros(20)], -1)
    np.testing.assert_allclose(first_observation(path)[:, 0], np.linspace(3, 7, 24), atol=1e-9)


def test_to_goal_cuts_at_first_crossing_with_interpolated_endpoint():
    path = np.array([[0.0, 0.0], [0.0, GOAL_Y - 0.1], [0.2, GOAL_Y + 0.1], [0.0, GOAL_Y - 0.1], [0.2, GOAL_Y + 0.2]])
    np.testing.assert_allclose(to_goal(path), [[0.0, 0.0], [0.0, GOAL_Y - 0.1], [0.1, GOAL_Y]], atol=1e-9)


def test_to_goal_leaves_a_path_that_never_arrives():
    path = np.array([[0.0, 0.0], [0.0, 0.1]])
    np.testing.assert_array_equal(to_goal(path), path)


def _routes():
    counts = [3, *[4] * 22, 5]
    return np.repeat(np.arange(24), counts)


def test_make_folds_is_stratified_partition():
    routes = _routes()
    folds = make_folds(routes, 0)
    assert np.bincount(folds).tolist() == [32, 32, 32]
    for r in range(24):
        per_fold = np.bincount(folds[routes == r], minlength=3)
        assert per_fold.max() - per_fold.min() <= 1
    np.testing.assert_array_equal(folds, make_folds(routes, 0))
    assert not np.array_equal(folds, make_folds(routes, 1))


def test_missing_npz_raises_with_fetch_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_d3il"):
        D3ILAvoiding("train", 0, root=str(tmp_path))


def test_dev_split_raises(tmp_path):
    with pytest.raises(ValueError, match="dev"):
        D3ILAvoiding("dev", 0, root=str(tmp_path))


def test_fold_out_of_range_raises(tmp_path):
    with pytest.raises(ValueError, match="fold"):
        D3ILAvoiding("train", 3, root=str(tmp_path))


@pytest.fixture(scope="module")
def all_items():
    train, test = D3ILAvoiding("train", 0), D3ILAvoiding("test", 0)
    return [x[i] for x in (train, test) for i in range(len(x))]


@needs_data
def test_96_demos_common_start():
    paths = demos(0, "train") + demos(0, "test")
    assert len(paths) == 96
    np.testing.assert_allclose([p[0] for p in paths], np.tile(START_POINT, (96, 1)), atol=1e-6)


@needs_data
def test_segment_count(all_items):
    assert len(all_items) == EXPECTED_SEGMENTS


@needs_data
def test_non_final_arc_length_along_the_recording():
    for path in _load(str(ROOT))[0]:
        recording = to_goal(path)[7:]
        pts, lengths = segments(recording)
        np.testing.assert_allclose(lengths[:-1], 0.08, rtol=1e-3)
        steps = np.linalg.norm(np.diff(recording, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(steps)])
        at = 0.08 * np.arange(1, len(pts))
        ends = np.stack([np.interp(at, arc, recording[:, axis]) for axis in range(2)], -1)
        np.testing.assert_allclose(pts[:-1, -1], ends, atol=1e-9)


@needs_data
def test_resampled_polyline_only_cuts_corners(all_items):
    for item in all_items:
        if item["fut_len"] < 0.0799:  # final segment
            continue
        points = torch.cat([item["obs_pts"][-1:], item["fut_pts"]]).double() * 0.4
        length = (points[1:] - points[:-1]).norm(dim=-1).sum().item()
        assert 0.08 * 0.98 < length < 0.08 * 1.001


@needs_data
def test_route_distribution(all_items):
    routes = {(int(x["demo_id"]), int(x["route"])) for x in all_items}
    assert sorted(Counter(r for _, r in routes).values()) == [3, *[4] * 22, 5]


@needs_data
def test_folds_disjoint_and_complete():
    ids = []
    for fold in range(3):
        train = {int(x["demo_id"]) for x in D3ILAvoiding("train", fold)}
        test = {int(x["demo_id"]) for x in D3ILAvoiding("test", fold)}
        assert (len(train), len(test)) == (64, 32) and not train & test
        ids += test
    assert sorted(ids) == list(range(96))


@needs_data
def test_observation_is_preceding_segment():
    ds = D3ILAvoiding("train", 0)
    for prev, item in zip(ds, list(ds)[1:], strict=False):
        if item["seg_idx"] > 0:
            assert torch.equal(item["obs_pts"], prev["fut_pts"])


@needs_data
def test_first_item_observation_is_interpolated_frames():
    paths, _ = _load(str(ROOT))
    ds = D3ILAvoiding("train", 0)
    first = next(x for x in ds if x["seg_idx"] == 0)
    expected = normalize(first_observation(paths[int(first["demo_id"])]))
    np.testing.assert_allclose(first["obs_pts"].numpy(), expected, atol=1e-5)


@needs_data
def test_every_demo_fits_the_window():
    points = normalize(np.concatenate(demos(0, "train") + demos(0, "test")))
    assert points[:, 0].min() >= WINDOW_X[0] and points[:, 0].max() <= WINDOW_X[1]
    assert points[:, 1].min() >= WINDOW_Y[0] and points[:, 1].max() <= WINDOW_Y[1]


@needs_data
def test_item_keys_and_shapes():
    item = D3ILAvoiding("train", 0)[0]
    assert set(item) == {"obs_pts", "fut_pts", "fut_len", "demo_id", "seg_idx", "route"}
    assert item["obs_pts"].shape == item["fut_pts"].shape == (24, 2)
    assert item["obs_pts"].dtype == item["fut_pts"].dtype == torch.float32
    assert all(item[k].shape == () for k in ("fut_len", "demo_id", "seg_idx", "route"))
