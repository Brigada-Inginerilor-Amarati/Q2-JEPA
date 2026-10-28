"""Pick the CUDA-Q wheel matching the Colab GPU driver (pyproject pins the CUDA-13 build).

python scripts/colab_setup.py
"""

import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

CUDAQ_VERSION = "0.16.0"


def cuda_major(nvidia_smi_output: str) -> int:
    match = re.search(r"CUDA Version:\s*(\d+)\.", nvidia_smi_output)
    if not match:
        raise RuntimeError("no 'CUDA Version' in nvidia-smi output; is a GPU runtime attached?")
    return int(match.group(1))


def cudaq_wheel(cuda_major: int) -> str:
    if cuda_major >= 13:
        return f"cuda-quantum-cu13=={CUDAQ_VERSION}"
    if cuda_major == 12:
        return f"cuda-quantum-cu12=={CUDAQ_VERSION}"
    raise RuntimeError(f"CUDA {cuda_major} driver is too old for CUDA-Q {CUDAQ_VERSION} (needs 12+)")


def main() -> None:
    wheel = cudaq_wheel(cuda_major(subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True).stdout))
    name = wheel.split("==")[0]
    try:
        installed = version(name) == CUDAQ_VERSION
    except PackageNotFoundError:
        installed = False
    if not installed:
        pip = [sys.executable, "-m", "pip"]
        subprocess.run([*pip, "uninstall", "-y", "cuda-quantum-cu13", "cuda-quantum-cu12"], check=True)
        subprocess.run([*pip, "install", "-q", wheel], check=True)
    print(f"CUDA-Q wheel: {wheel}")


if __name__ == "__main__":
    main()
