"""HQViT timing gate (phase-2 spec §6): seconds per training step and estimated hours per full run.

Times `q2jepa.train.train_step`, the function `run()` uses, on the phase-2 grid settings (`+experiment=phase2`:
64x64, 8x8 patches, batch 64). Per encoder x prior it measures a representation-stage step, a full-stage step
(prior sample + update, the most expensive kind) and the readout stage's no-grad encoding of obs and fut, then
estimates a whole run as warmup_repr x repr + (iters - warmup_repr) x full + readout.iters x readout. Prediction-
stage steps are counted as full ones, so the estimate is an upper bound; evaluation is left out (one pass).

uv run python scripts/bench_encoder_step.py --target nvidia --encoder cnn --encoder vit --encoder hqvit \
    --prior softmax_categorical --prior entangled_qcbm --cu-per-hour 1.8
"""

import argparse
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

import q2jepa.config  # noqa: F401  (registers the schema)
from q2jepa.benchmarks import build_benchmark
from q2jepa.quantum import get_target, set_target
from q2jepa.train import build_models, resolve_device, train_step, use_hqvit_backend

CONF = str(Path(__file__).resolve().parents[1] / "conf")


def estimate_hours(repr_s: float, full_s: float, readout_s: float, iters: int, warmup_repr: int,
                   readout_iters: int) -> float:  # fmt: skip
    return (warmup_repr * repr_s + (iters - warmup_repr) * full_s + readout_iters * readout_s) / 3600


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, steps: int, device: torch.device) -> float:
    fn()  # warm-up: CUDA-Q kernel JIT, cuDNN autotuning
    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / steps


def bench_step(
    target: str, encoder: str, prior: str, steps: int = 3, overrides: list[str] = (), experiment: str = "phase2"
) -> dict:
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[f"+experiment={experiment}", f"encoder={encoder}", f"prior={prior}", f"quantum.target={target}",
                       *overrides],
        )  # fmt: skip
    set_target(cfg.quantum.target, cfg.quantum.precision)
    device = resolve_device(cfg.trainer.device)
    groups, opt = build_models(cfg, device)
    enc, t, e = groups["encoder"][0], cfg.trainer, cfg.encoder
    bench = build_benchmark(cfg, device)
    hqvit_train = cfg.quantum.hqvit_train if encoder == "hqvit" else "cudaq"  # override applies to hqvit only
    use_hqvit_backend(enc, hqvit_train)  # as in run(): readout timing below uses it too
    torch.manual_seed(cfg.seed)
    batch = bench.batch(t.batch)

    def readout_encode() -> None:
        with torch.no_grad():
            bench.encode(enc, batch)

    repr_s = _time(lambda: train_step(cfg, groups, opt, bench, batch, 1), steps, device)
    full_s = _time(lambda: train_step(cfg, groups, opt, bench, batch, t.warmup_pred + 1), steps, device)
    readout_s = _time(readout_encode, steps, device)
    return {
        "encoder": encoder,
        "prior": prior,
        "target": get_target(),
        "hqvit_train": hqvit_train,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "batch": t.batch,
        "res": e.res,
        "repr_s": repr_s,
        "full_s": full_s,
        "readout_s": readout_s,
        "est_hours": estimate_hours(repr_s, full_s, readout_s, t.iters, t.warmup_repr, cfg.readout.iters),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", default="qpp-cpu")
    ap.add_argument("--encoder", action="append", help="cnn | vit | hqvit (repeatable; default all three)")
    ap.add_argument("--prior", action="append", help="prior config name (repeatable; default softmax_categorical)")
    ap.add_argument("--steps", type=int, default=3, help="timed steps per measurement, after one warm-up step")
    ap.add_argument("--cu-per-hour", type=float, default=0.0, help="Colab compute units per hour for this GPU")
    ap.add_argument("--experiment", default="phase2", help="preset: phase2 (MovingMNIST), hep18 or d3il")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra Hydra override")
    args = ap.parse_args()
    cu = args.cu_per_hour
    print("| encoder | prior | target | hqvit_train | device | batch | repr s/it | full s/it | readout s/it | est. h/run |"
          + (" est. CU/run |" if cu else ""))  # fmt: skip
    print("|---" * (10 + bool(cu)) + "|")
    for enc in args.encoder or ["cnn", "vit", "hqvit"]:
        for prior in args.prior or ["softmax_categorical"]:
            r = bench_step(args.target, enc, prior, args.steps, args.set, args.experiment)
            print(f"| {r['encoder']} | {r['prior']} | {r['target']} | {r['hqvit_train']} | {r['device']} | {r['batch']} | "
                  f"{r['repr_s']:.3f} | {r['full_s']:.3f} | {r['readout_s']:.3f} | {r['est_hours']:.2f} |"
                  + (f" {r['est_hours'] * cu:.1f} |" if cu else ""), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
