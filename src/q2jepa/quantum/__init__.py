import cudaq

PRECISIONS = ("default", "fp32", "fp64")


def set_target(name: str, precision: str = "default") -> None:
    """Select the CUDA-Q target. `precision` "fp32"/"fp64" picks the `nvidia` target's simulation precision
    (its own default is fp32); "default" keeps whatever the target uses (`qpp-cpu` is always fp64)."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")
    if precision == "default":
        cudaq.set_target(name)
        return
    cudaq.set_target(name, option=precision)
    if is_fp32() != (precision == "fp32"):  # a target that ignores the option would mislabel the run's platform
        raise ValueError(f"target {name!r} did not switch to {precision}; precision applies to nvidia only")


def get_target() -> str:
    return cudaq.get_target().name


def is_fp32() -> bool:
    """True when the active target simulates in single precision (the `nvidia` default)."""
    return cudaq.get_target().get_precision() == cudaq.SimulationPrecision.fp32
