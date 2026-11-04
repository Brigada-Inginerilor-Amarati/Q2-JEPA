"""Top-down videos for people: HEP18's 18 futures and D3IL's 24 routes, with roads, obstacles and the goal line.

These are not encoder inputs (see render_agent_clip and render_segment). Needs ffmpeg on the PATH.

uv run --with matplotlib python scripts/render_videos.py --out outputs/render_videos
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

from q2jepa.data import d3il, hep18

FPS = 20
HOLD = FPS  # frames to pause on the final state
PRIORITY, YIELDING, OBSERVED = "#d9480f", "#1971c2", "#343a40"


def _draw_road(ax, view: float) -> None:
    half = hep18.LANE_WIDTH  # one lane each way
    ax.add_patch(Rectangle((-view, -half), 2 * view, 2 * half, color="#dee2e6", zorder=0))
    ax.add_patch(Rectangle((-half, -view), 2 * half, 2 * view, color="#dee2e6", zorder=0))
    box = hep18.INTERSECTION_HALF
    ax.add_patch(Rectangle((-box, -box), 2 * box, 2 * box, fill=False, ec="#adb5bd", ls=":", lw=0.8))
    ax.set(xlim=(-view, view), ylim=(view, -view), aspect="equal", xticks=[], yticks=[])  # +y is south


def hep18_video(path: Path, obs: int = 0) -> None:
    """One train observation's 18 futures in real time. Rows: first priority car's schedule; columns: axis, second car's."""
    ds = hep18.HEP18("train")
    d0, v0 = ds.d0[obs], ds.v0[obs]
    t = np.arange(-2.0, 9.0 + 1e-9, 1 / FPS)
    schedules = np.stack([hep18.future_schedules(j) for j in range(18)])
    s, _ = hep18.car_state(d0, v0, hep18.schedule_acc(schedules, v0), t)  # (18, 4, T)
    xy = hep18.LANE_POINT[:, None] + s[..., None] * hep18.HEADING[:, None]  # (18, 4, T, 2)
    half = hep18._HALF_EXTENT

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    cars = []
    for j in range(18):
        axis, first, second = j // 9, (j % 9) // 3, j % 3
        ax = axes[first, 3 * axis + second]
        _draw_road(ax, view=68.0)
        ax.set_title(f"{('NS', 'EW')[axis]} priority: {'ABC'[first]}, {'ABC'[second]}", fontsize=9)
        cars.append([ax.add_patch(Rectangle((0, 0), *(2 * half[c]))) for c in range(4)])

    def update(frame: int) -> None:
        i = min(frame, len(t) - 1)
        phase = "observation" if t[i] <= 0 else "future"
        fig.suptitle(f"HEP18, train observation {obs}: t = {t[i]:+.2f} s ({phase})")
        for j, rects in enumerate(cars):
            for c, rect in enumerate(rects):
                rect.set_xy(xy[j, c, i] - half[c])
                colour = YIELDING if schedules[j, c] == hep18.SCHEDULE_D else PRIORITY
                rect.set_color(OBSERVED if t[i] <= 0 else colour)

    FuncAnimation(fig, update, frames=len(t) + HOLD).save(path, writer="ffmpeg", fps=FPS, dpi=100)
    plt.close(fig)


def d3il_video(path: Path) -> None:
    """One fold-0 train demo per route, one recorded frame per video frame. Eval demos stay out of this."""
    by_route = {}
    for demo in d3il.demos(0, "train"):
        by_route.setdefault(d3il.route_label(demo), demo)
    x_window = np.array(d3il.WINDOW_X) * d3il.NORM_SCALE + d3il.NORM_CENTRE[0]  # the encoder's view, in metres
    y_window = np.array(d3il.WINDOW_Y) * d3il.NORM_SCALE + d3il.NORM_CENTRE[1]

    fig, axes = plt.subplots(4, 6, figsize=(15, 11))
    trails, heads = [], []
    for ax, route in zip(axes.flat, sorted(by_route), strict=True):
        for row_y, xs, radius in zip(d3il.ROW_Y, d3il.ROW_OBSTACLE_X, d3il.ROW_RADIUS, strict=True):
            for x in xs:
                ax.add_patch(Circle((x, row_y), radius, color="#868e96"))
        ax.axhline(d3il.GOAL_Y, color="#2f9e44", lw=1)
        ax.set(xlim=x_window, ylim=y_window, aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"route {route}", fontsize=9)
        trails.append(ax.plot([], [], color=PRIORITY, lw=1.2)[0])
        heads.append(ax.plot([], [], "o", color=YIELDING, ms=4)[0])
    paths = [by_route[r] for r in sorted(by_route)]

    def update(frame: int) -> None:
        fig.suptitle(f"D3IL Avoiding, fold 0 train: recorded frame {min(frame, max(map(len, paths)) - 1)}")
        for p, trail, head in zip(paths, trails, heads, strict=True):
            seen = p[: frame + 1]
            trail.set_data(seen[:, 0], seen[:, 1])
            head.set_data(seen[-1:, 0], seen[-1:, 1])

    frames = max(map(len, paths)) + HOLD
    FuncAnimation(fig, update, frames=frames).save(path, writer="ffmpeg", fps=FPS, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/render_videos"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    hep18_video(args.out / "hep18.mp4")
    d3il_video(args.out / "d3il.mp4")
    print(f"saved to {args.out}")
