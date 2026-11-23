"""Grid launcher: run counts, schedule selection, priority order, skip-and-resume."""

import json

import pytest

from scripts import hep18_fix_sweep, run_grid
from tests.test_train import TINY


def _tuned(root, bench, scores):
    """Fake finished tune runs: scores[encoder] = [(coverage, tv) for s1, s2, s3]."""
    for enc, per_sched in scores.items():
        for s, (cov, tv) in zip(run_grid.SCHEDULES, per_sched, strict=True):
            d = root / bench / "tune" / f"{enc}-{s}"
            d.mkdir(parents=True)
            (d / "metrics_dev.json").write_text(json.dumps({"coverage": cov, "tv": tv}))


ALL_EQUAL = {e: [(0.5, 0.2)] * 3 for e in run_grid.ENCODERS}


def test_run_counts(tmp_path):
    assert len(run_grid.plan("mmnist", "tune", tmp_path)) == 9
    _tuned(tmp_path, "mmnist", ALL_EQUAL)
    _tuned(tmp_path, "hep18", ALL_EQUAL)
    assert len(run_grid.plan("mmnist", "grid", tmp_path)) == 18
    assert len(run_grid.plan("hep18", "grid", tmp_path)) == 21  # + Product-QCBM for HQViT
    d3il = run_grid.plan("d3il", "grid", tmp_path)  # tuned on MovingMNIST
    assert len(d3il) == 54 and len({j.name for j in d3il}) == 54
    with pytest.raises(SystemExit, match="reuses MovingMNIST"):
        run_grid.plan("d3il", "tune", tmp_path)
    assert len(run_grid.plan("nri", "grid", tmp_path)) == 18  # fast iteration: MovingMNIST's schedules
    assert len(run_grid.plan("nri", "grid", tmp_path, seeds=(0,))) == 6
    with pytest.raises(SystemExit, match="reuses MovingMNIST"):
        run_grid.plan("nri", "tune", tmp_path)
    tune = run_grid.plan("chess", "tune", tmp_path)  # chess has its own tuning (spec §4.3)
    assert len(tune) == 9 and all("+experiment=chess" in j.overrides for j in tune)
    with pytest.raises(SystemExit, match="chess tuning"):
        run_grid.plan("chess", "grid", tmp_path)
    _tuned(tmp_path, "chess", ALL_EQUAL)
    assert len(run_grid.plan("chess", "grid", tmp_path, seeds=(0,))) == 6


def test_best_schedule_and_priority(tmp_path):
    _tuned(tmp_path, "mmnist", {**ALL_EQUAL, "vit": [(0.5, 0.2), (0.6, 0.3), (0.6, 0.25)]})
    assert run_grid.best_schedule(tmp_path, "mmnist", "vit") == "s3"  # coverage tie: lower TV wins
    assert run_grid.best_schedule(tmp_path, "d3il", "vit") == "s3"  # D3IL reuses MovingMNIST's tuning
    jobs = run_grid.plan("mmnist", "grid", tmp_path)
    assert [j.name for j in jobs[:6]] == [f"{e}-entangled_qcbm-{s}-s{i}" for e, s in (("hqvit", "s1"), ("vit", "s3"))
                                          for i in range(3)]  # fmt: skip
    assert "quantum.hqvit_train=torch" in jobs[0].overrides and "eval.split=test" in jobs[0].overrides


def test_tuning_from_another_account(tmp_path):
    _tuned(tmp_path / "daria", "mmnist", {**ALL_EQUAL, "cnn": [(0.5, 0.2), (0.9, 0.2), (0.5, 0.2)]})
    jobs = run_grid.plan("d3il", "grid", tmp_path / "paul", tune_root=tmp_path / "daria")
    assert {j.name.split("-")[2] for j in jobs if j.name.startswith("cnn")} == {"s2"}


def test_grid_needs_finished_tuning(tmp_path):
    with pytest.raises(SystemExit, match="unfinished"):
        run_grid.plan("hep18", "grid", tmp_path)


def test_runs_skip_finished(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(run_grid, "GPU", ["trainer.device=cpu"])  # launcher logic only; real runs are GPU-only
    args = ["--bench", "mmnist", "--stage", "tune", "--root", str(tmp_path), "--encoders", "cnn", "--jobs", "3",
            *[a for o in TINY for a in ("--set", o)]]  # fmt: skip
    assert run_grid.main(args) == 0
    assert all((tmp_path / "mmnist" / "tune" / f"cnn-{s}" / "metrics_dev.json").exists() for s in run_grid.SCHEDULES)
    assert run_grid.main(args) == 0
    assert "3 finished, 0 to run" in capsys.readouterr().out
    progress = (tmp_path / "mmnist" / "tune" / "progress.md").read_text()
    assert (
        progress.startswith("# mmnist tune: 3 done, 0 running, 0 failed, 0 pending (")
        and "| cnn-s2 | done |" in progress
    )
    assert run_grid.last_step(tmp_path / "mmnist" / "tune" / "cnn-s1") == "evaluating"  # last line: final metrics
    cfg = (tmp_path / "mmnist" / "tune" / "cnn-s1" / ".hydra" / "config.yaml").read_text()
    assert "name: cnn-s1" in cfg and "group: mmnist-tune" in cfg


def test_images_only():
    """Grid, tuning and the HEP18 fix sweep never use the diagnostic state encoder (AGENTS.md §3)."""
    assert "state" not in run_grid.ENCODERS
    assert not any(o == "encoder=state" for runs in hep18_fix_sweep.RUNS.values() for o in runs)
    assert all(any(o.startswith("seed=") for o in runs) for runs in hep18_fix_sweep.RUNS.values())


def test_shared_root_guards(tmp_path):
    with pytest.raises(SystemExit):  # argparse error: no --cpu flag any more
        run_grid.main(["--bench", "mmnist", "--stage", "tune", "--cpu", "--dry-run"])
    with pytest.raises(SystemExit):  # --set without --root
        run_grid.main(["--bench", "mmnist", "--stage", "tune", "--set", "trainer.iters=5", "--dry-run"])


def test_resume_refuses_another_commit(tmp_path):
    run = tmp_path / "mmnist" / "tune" / "cnn-s1"
    run.mkdir(parents=True)
    (run / "commit.txt").write_text("0" * 40 + "\n")
    with pytest.raises(SystemExit, match="another commit"):
        run_grid.main(["--bench", "mmnist", "--stage", "tune", "--root", str(tmp_path), "--encoders", "cnn"])


def test_progress_survives_partial_files(tmp_path):
    """A run mid-write (half a log line, an empty metrics file) must not crash the launcher."""
    jobs = run_grid.plan("mmnist", "tune", tmp_path, encoders=("cnn",))
    base = tmp_path / "mmnist" / "tune"
    (base / "cnn-s1").mkdir(parents=True)
    (base / "cnn-s1" / "log.jsonl").write_text('{"it": 5, "loss": 0.5}\n{"it": 6, "lo')
    (base / "cnn-s2").mkdir()
    (base / "cnn-s2" / "metrics_dev.json").write_text("")
    run_grid.write_progress(base, jobs, {"cnn-s1", "cnn-s2"}, set())
    progress = (base / "progress.md").read_text()
    assert "| cnn-s1 | running | it 5, loss 0.5 |" in progress and "| cnn-s2 | running |" in progress


def _hydra_pool(run_dir, pool):
    (run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    (run_dir / ".hydra" / "config.yaml").write_text(f"encoder:\n  name: vit\n  pool: {pool}\n")


def test_bench_preset_pooling(tmp_path):
    # conf/experiment/nri.yaml sets encoder.pool; nri still reuses MovingMNIST's (mean-pooled) tune runs.
    assert run_grid.default_pool("nri") == "flatten" and run_grid.default_pool("chess") == "mean"
    _tuned(tmp_path, "mmnist", ALL_EQUAL)
    for s in run_grid.SCHEDULES:
        _hydra_pool(tmp_path / "mmnist" / "tune" / f"vit-{s}", "mean")
    assert len(run_grid.plan("nri", "grid", tmp_path, pool="flatten")) == 18


def test_stale_pooling_is_refused(tmp_path, monkeypatch):
    # phase-2 spec §5.6: twin/HQViT tune runs trained with another pooling must not pick the grid's schedule.
    assert run_grid.default_pool() == "mean"
    _tuned(tmp_path, "mmnist", ALL_EQUAL)
    for s in run_grid.SCHEDULES:
        _hydra_pool(tmp_path / "mmnist" / "tune" / f"vit-{s}", "softargmax")
        _hydra_pool(tmp_path / "mmnist" / "tune" / f"cnn-{s}", "softargmax")  # CNN ignores pooling
    with pytest.raises(SystemExit, match="encoder.pool=softargmax"):
        run_grid.plan("mmnist", "grid", tmp_path)
    assert run_grid.best_schedule(tmp_path, "mmnist", "cnn") == "s1"
    assert len(run_grid.plan("mmnist", "grid", tmp_path, pool="softargmax")) == 18  # the ablation root
    monkeypatch.setattr(run_grid, "git_sha", lambda: "abc")
    with pytest.raises(SystemExit, match="tune-softargmaxpool"):
        run_grid.main(["--bench", "mmnist", "--stage", "tune", "--root", str(tmp_path), "--dry-run"])
    for s in run_grid.SCHEDULES:
        _hydra_pool(tmp_path / "mmnist" / "tune" / f"vit-{s}", "mean")
    assert len(run_grid.plan("mmnist", "grid", tmp_path)) == 18
