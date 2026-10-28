import cudaq

from q2jepa.quantum import set_target


def test_cudaq_target_selected():
    set_target("qpp-cpu")
    assert cudaq.get_target().name == "qpp-cpu"
