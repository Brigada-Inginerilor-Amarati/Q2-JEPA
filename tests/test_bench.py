import pytest

from scripts.bench_attention import run_bench
from scripts.bench_encoder_step import bench_step, estimate_hours

TINY = ["trainer.batch=2", "trainer.k_train=2", "dataset.n_train=2", "encoder.res=16"]


def test_bench_rows():
    rows = run_bench("qpp-cpu", 16, 1)
    assert {r["mode"] for r in rows} == {"factored", "full"}
    assert {r["qubits"] for r in rows} == {4, 17} and all(r["ms_per_image"] > 0 for r in rows)


def test_estimate_hours_counts_every_stage():
    # 200 repr iterations at 1 s, 3800 full at 2 s, 4000 readout iterations at 0.5 s
    assert estimate_hours(1.0, 2.0, 0.5, iters=4000, warmup_repr=200, readout_iters=4000) == pytest.approx(
        (200 + 7600 + 2000) / 3600
    )


@pytest.mark.parametrize(
    ("encoder", "prior", "extra"),
    [
        ("cnn", "softmax_categorical", []),
        ("cnn", "entangled_qcbm", ["quantum.shots=8", "quantum.shift_obs=2", "quantum.shift_coords=3"]),
        ("hqvit", "softmax_categorical", ["encoder.patch=4", "encoder.depth=1"]),
    ],
    ids=["cnn", "qcbm", "hqvit"],
)
def test_bench_step_row(encoder, prior, extra):
    row = bench_step("qpp-cpu", encoder, prior, steps=1, overrides=TINY + extra)
    assert (row["encoder"], row["prior"], row["target"]) == (encoder, prior, "qpp-cpu")
    assert all(row[k] > 0 for k in ("repr_s", "full_s", "readout_s", "est_hours"))
    assert row["batch"] == 2 and row["res"] == 16  # overrides reach the timed models
    assert row["est_hours"] == pytest.approx(
        estimate_hours(row["repr_s"], row["full_s"], row["readout_s"], 4000, 200, 4000)
    )
