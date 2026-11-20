"""Save example clips, print mean frame norms and renderer timings. Rerun whenever SIGMA changes.

uv run python scripts/render_examples.py --out outputs/render_examples
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torchvision.utils import save_image

from q2jepa.data.d3il import D3ILAvoiding, render_segment
from q2jepa.data.hep18 import HEP18, render_agent_clip


def mean_frame_norm(clips: torch.Tensor) -> float:
    return clips.norm(dim=(-2, -1)).mean().item()


def _all_frames_norm(render: Callable, states: torch.Tensor, batch: int = 1024) -> float:
    total = sum(
        mean_frame_norm(render(states[i : i + batch])) * len(states[i : i + batch])
        for i in range(0, len(states), batch)
    )
    return total / len(states)


def _report(name: str, norm: float) -> None:
    print(f"{name}: mean frame L2 norm {norm:.4f}, encoder.input_scale = {norm / 64:.4f}")


def hep18_examples(out: Path) -> None:
    ds = HEP18("train")
    obs = torch.tensor(ds.obs_state[..., 0], dtype=torch.float32).reshape(-1, 3)
    fut = torch.tensor(ds.fut_state[..., 0], dtype=torch.float32).reshape(-1, 3)
    _report("hep18", _all_frames_norm(render_agent_clip, torch.cat([obs, fut])))
    # rows: four cars' observation, then their futures under joint future 0 (NS priority, both schedule A)
    grid = torch.cat([render_agent_clip(obs[:4]), render_agent_clip(fut[:4])]).reshape(-1, 1, 64, 64)
    save_image(grid, out / "hep18.png", nrow=12, pad_value=0.5)


def d3il_examples(out: Path) -> None:
    ds = D3ILAvoiding("train", 0)
    items = [ds[i] for i in range(len(ds))]
    states = torch.stack([x[k] for x in items for k in ("obs_pts", "fut_pts")])
    _report("d3il", _all_frames_norm(render_segment, states, batch=256))
    stride = max(len(items) // 16, 1)
    grid = render_segment(torch.stack([x["fut_pts"] for x in items[::stride]][:16])).reshape(-1, 1, 64, 64)
    save_image(grid, out / "d3il.png", nrow=12, pad_value=0.5)
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        batch = states[:384].to(device)
        start = time.perf_counter()
        render_segment(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"render_segment, batch of 384 on {device}: {(time.perf_counter() - start) * 1e3:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/render_examples"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hep18_examples(args.out)
    try:
        d3il_examples(args.out)
    except FileNotFoundError as e:
        print(f"skipping d3il: {e}")
    print(f"saved to {args.out}")
