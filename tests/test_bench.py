from scripts.bench_attention import run_bench


def test_bench_rows():
    rows = run_bench("qpp-cpu", 16, 1)
    assert {r["mode"] for r in rows} == {"factored", "full"}
    assert {r["qubits"] for r in rows} == {4, 17} and all(r["ms_per_image"] > 0 for r in rows)
