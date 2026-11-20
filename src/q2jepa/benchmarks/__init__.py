"""One module per benchmark: training batches, scene encoding, readout fitting and evaluation. train.py has the loop.

Interface (duck-typed):
- batch(size) -> batch, drawn with the global torch RNG (train.py reseeds per iteration);
- encode(encoder, batch) -> h (B, d), targets (B, d) or (B, J, d) for J futures per observation;
- fit_readout(cfg, encoder, out_dir, log) -> readout (loads readout.pt if present, else trains and saves it);
- post_readout(cfg, groups, readout, out_dir, log) -> None (D3IL's final predictor/prior stage; others: no-op);
- evaluate(cfg, encoder, predictor, prior, readout) -> metrics dict.
"""


def build_benchmark(cfg, device):
    name = cfg.dataset.name
    if name == "mmnist":
        from q2jepa.benchmarks.mmnist import MovingMNISTBenchmark

        return MovingMNISTBenchmark(cfg, device)
    if name == "hep18":
        from q2jepa.benchmarks.hep18 import HEP18Benchmark

        return HEP18Benchmark(cfg, device)
    if name == "d3il":
        from q2jepa.benchmarks.d3il import D3ILBenchmark

        return D3ILBenchmark(cfg, device)
    raise ValueError(f"unknown dataset {name!r}")


def scene_dim(cfg) -> int:
    """Width of h: the encoder's output, times the number of agents encoded separately (HEP18: 4)."""
    return cfg.encoder.d * cfg.dataset.agents
