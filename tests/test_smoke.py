import pytest

from q2jepa.quantum import get_target, set_target


def test_cudaq_target_selected():
    set_target("qpp-cpu")
    assert get_target() == "qpp-cpu"


def test_unknown_precision_is_rejected_before_switching_target():
    with pytest.raises(ValueError, match="precision"):
        set_target("qpp-cpu", "fp16")
