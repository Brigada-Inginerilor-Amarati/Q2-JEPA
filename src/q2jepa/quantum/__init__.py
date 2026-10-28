import cudaq


def set_target(name: str) -> None:
    cudaq.set_target(name)


def get_target() -> str:
    return cudaq.get_target().name
