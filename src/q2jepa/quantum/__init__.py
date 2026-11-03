import cudaq


def set_target(name: str) -> None:
    cudaq.set_target(name)


def get_target() -> str:
    return cudaq.get_target().name


def is_fp32() -> bool:
    """True when the active target simulates in single precision (the `nvidia` default)."""
    return cudaq.get_target().get_precision() == cudaq.SimulationPrecision.fp32
