import numpy as np
import pytest
import torch

from q2jepa.data.d3il import START_POINT, WINDOW_X, WINDOW_Y, normalize, render_segment, to_pixels


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
