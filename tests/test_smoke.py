from q2jepa.quantum import get_target, set_target


def test_cudaq_target_selected():
    set_target("qpp-cpu")
    assert get_target() == "qpp-cpu"
