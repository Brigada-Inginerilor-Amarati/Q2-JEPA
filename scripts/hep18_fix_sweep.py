"""HEP18 fix sweep (runbook §1a): 10 dev runs on rendered images on a T4, then a summary table.

python scripts/hep18_fix_sweep.py --jobs 3            # rerun after a disconnect: finished runs are skipped
python scripts/hep18_fix_sweep.py --dry-run           # print the commands only
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_ROOT = "/content/drive/MyDrive/q2jepa/runs/hep18/fix"
BASE = ["+experiment=hep18", "eval.split=dev", "quantum=gpu", "trainer.device=cuda", "wandb.mode=disabled"]
RUNS = {  # rendered images only (AGENTS.md §3); prior settings on the CNN, encoder learning rate on CNN and twin
    "cnn-mix": ["encoder=cnn", "prior=mixture", "seed=0"],
    "cnn-lrp": ["encoder=cnn", "trainer.lr_prior=3e-3", "seed=0"],
    "cnn-long": ["encoder=cnn", "trainer.iters=12000", "seed=0"],
    "cnn-lre-hi": ["encoder=cnn", "trainer.lr_enc=1e-3", "seed=0"],
    "cnn-lre-lo": ["encoder=cnn", "trainer.lr_enc=1e-4", "seed=0"],
    "cnn-t2": ["encoder=cnn", "+tune=s2", "seed=0"],
    "cnn-t3": ["encoder=cnn", "+tune=s3", "seed=0"],
    "vit-lre-hi": ["encoder=vit", "trainer.lr_enc=1e-3", "seed=0"],
    "vit-t2": ["encoder=vit", "+tune=s2", "seed=0"],
    "vit-t3": ["encoder=vit", "+tune=s3", "seed=0"],
}
KEYS = ("validity", "coverage", "local_validity", "readout_validity", "tv")


def command(name: str, root: Path) -> list[str]:
    return [sys.executable, "-m", "q2jepa.train", *BASE, *RUNS[name], f"hydra.run.dir={root / name}"]


def run(name: str, root: Path) -> tuple[str, int]:
    (root / name).mkdir(parents=True, exist_ok=True)
    with open(root / name / "run.log", "a") as log:
        return name, subprocess.run(command(name, root), check=False, stdout=log, stderr=subprocess.STDOUT).returncode


def table(root: Path) -> str:
    rows = [" | ".join(["run", *KEYS])]
    for name in RUNS:
        path = root / name / "metrics_dev.json"
        m = json.loads(path.read_text()) if path.exists() else {}
        rows.append(" | ".join([name, *(f"{m[k]:.3f}" if k in m else "-" for k in KEYS)]))
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    todo = [n for n in RUNS if not (args.root / n / "metrics_dev.json").exists()]
    print(f"{len(RUNS) - len(todo)} finished, {len(todo)} to run")
    if args.dry_run:
        for n in todo:
            print(" ".join(command(n, args.root)[1:]))
        return 0
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = dict(pool.map(lambda n: run(n, args.root), todo))
    failed = sorted(n for n, c in codes.items() if c)
    print(table(args.root))
    if failed:
        print(f"failed: {failed}; see <root>/<run>/run.log, then rerun this command")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
