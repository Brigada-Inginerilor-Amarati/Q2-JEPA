import pytest

from scripts.colab_setup import cuda_major, cudaq_wheel


def test_parse_and_pick():
    hdr = "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4     |"
    assert cuda_major(hdr) == 12 and cudaq_wheel(12) == "cuda-quantum-cu12==0.16.0"
    assert cudaq_wheel(13) == "cuda-quantum-cu13==0.16.0"
    with pytest.raises(RuntimeError):
        cudaq_wheel(11)
