"""Run a benchmark's tuning or grid stage (runbook: docs/superpowers/plans/2026-11-20-training-runbook.md).

Each run gets its own dir <root>/<bench>/<stage>/<name>. Finished runs (metrics present) are skipped and
interrupted ones resume from their checkpoint, so after a Colab disconnect just rerun the same command.

  tune: every encoder x schedules s1-s3, seed 0, Softmax-Categorical, dev split (phase-2 spec F5).
  grid: encoders x priors x seeds (x folds for D3IL), test split, each encoder on its best tuned schedule:
        best dev coverage (ties: lower TV) of its tune runs. D3IL has no dev split and uses MovingMNIST's.

python scripts/run_grid.py --bench mmnist --stage tune --jobs 3
python scripts/run_grid.py --bench mmnist --stage grid --jobs 3
python scripts/run_grid.py --bench hep18 --stage grid --dry-run     # print the commands only
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import NamedTuple

DEFAULT_ROOT = "/content/drive/MyDrive/q2jepa/runs"
REPO = Path(__file__).resolve().parents[1]
PRESET = {"mmnist": "+experiment=phase2", "hep18": "+experiment=hep18", "d3il": "+experiment=d3il"}
TUNED_ON = {"mmnist": "mmnist", "hep18": "hep18", "d3il": "mmnist"}  # D3IL has no dev split (spec §5.6)
ENCODERS = ("cnn", "vit", "hqvit")
PRIORS = ("entangled_qcbm", "softmax_categorical")  # entangled first: the priority rows (spec §9)
SCHEDULES = ("s1", "s2", "s3")
SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2)
GPU = ["quantum=gpu", "trainer.device=cuda"]
PROGRESS_EVERY = 300  # seconds between progress.md refreshes
PRIORITY = {"hqvit": 0, "vit": 1}  # spec §9: HQViT vs twin with Entangled-QCBM first


class Job(NamedTuple):
    name: str
    overrides: list[str]
    metrics: str  # file that marks the run as finished
    priority: tuple = ()  # lower runs first


def encoder_overrides(encoder: str) -> list[str]:
    return [f"encoder={encoder}", *(["quantum.hqvit_train=torch"] if encoder == "hqvit" else [])]


def best_schedule(root: Path, bench: str, encoder: str) -> str:
    """The encoder's tune run with the best dev coverage (ties: lower TV)."""
    scores = {}
    for s in SCHEDULES:
        path = root / TUNED_ON[bench] / "tune" / f"{encoder}-{s}" / "metrics_dev.json"
        if path.exists():
            m = json.loads(path.read_text())
            scores[s] = (m["coverage"], -m["tv"])
    if len(scores) < len(SCHEDULES):
        missing = sorted(set(SCHEDULES) - set(scores))
        raise SystemExit(f"{TUNED_ON[bench]} tuning for {encoder} is unfinished (missing {missing}): run --stage tune")
    return max(scores, key=scores.get)


def plan(
    bench: str, stage: str, root: Path, encoders=ENCODERS, priors=PRIORS, seeds=SEEDS, tune_root: Path | None = None
) -> list[Job]:
    """Runs of one stage. Grid schedules come from the tune runs under tune_root (default: root)."""
    if stage == "tune":
        if bench == "d3il":
            raise SystemExit("D3IL has no dev split: it reuses MovingMNIST's tuning (run --bench mmnist --stage tune)")
        return [
            Job(f"{e}-{s}", [PRESET[bench], f"+tune={s}", *encoder_overrides(e), "seed=0"], "metrics_dev.json")
            for e in encoders
            for s in SCHEDULES
        ]
    jobs = []
    folds = FOLDS if bench == "d3il" else (None,)
    for e in encoders:
        sched = best_schedule(tune_root or root, bench, e)
        rows = list(priors) + (["product_qcbm"] if bench == "hep18" and e == "hqvit" else [])
        for p in rows:
            for seed in seeds:
                for f in folds:
                    name = f"{e}-{p}-{sched}-s{seed}" + ("" if f is None else f"-f{f}")
                    o = [PRESET[bench], f"+tune={sched}", "eval.split=test", *encoder_overrides(e), f"prior={p}",
                         f"seed={seed}", *([] if f is None else [f"dataset.fold={f}"])]  # fmt: skip
                    jobs.append(Job(name, o, "metrics.json", (p != "entangled_qcbm", PRIORITY.get(e, 2))))
    return sorted(jobs, key=lambda j: j.priority)  # stable: seeds and folds keep their order


def command(job: Job, run_dir: Path, platform: list[str], extra: list[str]) -> list[str]:
    bench, stage = run_dir.parent.parent.name, run_dir.parent.name
    tags = [f"wandb.name={job.name}", f"wandb.group={bench}-{stage}"]
    return [sys.executable, "-m", "q2jepa.train", *job.overrides, *platform, *tags, *extra, f"hydra.run.dir={run_dir}"]


def last_step(run_dir: Path) -> str:
    """The latest training step from <run dir>/log.jsonl, e.g. 'it 2350, loss 0.412'."""
    path = run_dir / "log.jsonl"
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    if not lines:
        return ""
    d = json.loads(lines[-1])
    for stage in ("it", "readout_it", "finetune_it"):  # world model, readout, D3IL stage 3
        if stage in d:
            loss = next((v for k, v in d.items() if k.endswith("loss")), None)
            return f"{stage} {d[stage]}" + (f", loss {loss:.4g}" if loss is not None else "")
    return "evaluating"


def write_progress(base: Path, jobs: list[Job], running: set[str], failed: set[str]) -> None:
    """<base>/progress.md: status, latest step and headline metrics per run; rewritten every few minutes."""
    keys = ("coverage", "validity", "tv", "acceptance")
    rows, counts = [], {"done": 0, "running": 0, "failed": 0, "pending": 0}
    for j in jobs:
        path = base / j.name / j.metrics
        m = json.loads(path.read_text()) if path.exists() else {}
        status = "done" if m else "failed" if j.name in failed else "running" if j.name in running else "pending"
        counts[status] += 1
        step = last_step(base / j.name) if status in ("running", "failed") else ""
        cells = [f"{m[k]:.3f}" if k in m else "" for k in keys]
        rows.append(f"| {j.name} | {status} | {step} | " + " | ".join(cells) + " |")
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    head = f"# {base.parent.name} {base.name}: " + ", ".join(f"{v} {k}" for k, v in counts.items()) + f" ({stamp})"
    table = ["| run | status | latest step | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 3), *rows]
    base.mkdir(parents=True, exist_ok=True)
    (base / "progress.md").write_text("\n".join([head, "", *table, ""]))


def git_sha() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def started_on(run_dir: Path) -> str | None:
    """The commit a run started on (first line of <run dir>/commit.txt), or None if it never started."""
    path = run_dir / "commit.txt"
    return path.read_text().split()[0] if path.exists() and path.read_text().strip() else None


def run_one(job: Job, run_dir: Path, platform: list[str], extra: list[str], started: set) -> tuple[str, int]:
    started.add(job.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "commit.txt", "a") as f:  # one line per session; resume refuses another commit (main)
        f.write(git_sha() + "\n")
    with open(run_dir / "run.log", "a") as log:
        cmd = command(job, run_dir, platform, extra)
        code = subprocess.run(cmd, check=False, stdout=log, stderr=subprocess.STDOUT).returncode
    return job.name, code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", required=True, choices=sorted(PRESET))
    ap.add_argument("--stage", required=True, choices=("tune", "grid"))
    ap.add_argument("--root", type=Path, default=None, help=f"run dirs go to <root>/<bench>/<stage>/ ({DEFAULT_ROOT})")
    ap.add_argument(
        "--tune-root",
        type=Path,
        default=None,
        help="where the tune runs live if another account ran them, e.g. a shared Drive folder",
    )
    ap.add_argument("--jobs", type=int, default=3, help="runs at once on the GPU")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=ENCODERS)
    ap.add_argument("--priors", nargs="+", default=list(PRIORS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument(
        "--wandb",
        default="offline",
        choices=("online", "offline", "disabled"),
        help="online: live curves in the team's wandb project (needs WANDB_API_KEY)",
    )
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra Hydra override (all runs)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args(argv)
    if args.root is None:
        if args.set:  # one-off settings stay out of the shared results folder (parity)
            ap.error("--set needs an explicit --root outside the shared results folder")
        args.root = Path(DEFAULT_ROOT)

    base = args.root / args.bench / args.stage
    jobs = plan(args.bench, args.stage, args.root, args.encoders, args.priors, args.seeds, args.tune_root)
    todo = [j for j in jobs if not (base / j.name / j.metrics).exists()]
    print(f"{args.bench} {args.stage}: {len(jobs)} runs, {len(jobs) - len(todo)} finished, {len(todo)} to run")
    sha = git_sha()
    moved = [j.name for j in todo if started_on(base / j.name) not in (None, sha)]
    if moved and not args.dry_run:  # a run must finish on the commit it started on (every row: overrides + seed + SHA)
        raise SystemExit(
            f"{moved} started on another commit than {sha}: check that commit out, or delete those run dirs"
        )
    platform = GPU  # all training runs on a Colab GPU (AGENTS.md §3)
    extra = [f"wandb.mode={args.wandb}", *args.set]
    if args.dry_run:
        for j in todo:
            print(" ".join(command(j, base / j.name, platform, extra)[1:]))
        return 0
    failed: set[str] = set()
    running: set[str] = set()  # filled by run_one as runs start
    write_progress(base, jobs, running, failed)
    with ThreadPoolExecutor(args.jobs) as pool:
        pending = {pool.submit(run_one, j, base / j.name, platform, extra, running) for j in todo}
        while pending:
            finished, pending = wait(pending, timeout=PROGRESS_EVERY, return_when=FIRST_COMPLETED)
            for fut in finished:
                name, code = fut.result()
                print(f"{'done' if code == 0 else f'FAILED ({code})'}: {name}", flush=True)
                running.discard(name)
                if code:
                    failed.add(name)
            write_progress(base, jobs, running, failed)
    if failed:
        print(f"{len(failed)} failed; see <run dir>/run.log, then rerun this command: {sorted(failed)}")
        return 1
    print(f"all {args.stage} runs finished. Summary: python scripts/plot_results.py {base} "
          f"--split {'dev' if args.stage == 'tune' else 'test'} --out {base}/results")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
