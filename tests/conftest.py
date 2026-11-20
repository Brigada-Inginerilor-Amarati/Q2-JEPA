import os

import pytest

from q2jepa.quantum import set_target


@pytest.fixture(scope="session", autouse=True)
def _cudaq_target():
    # Q2JEPA_CUDAQ_PRECISION=fp64 runs the nvidia target in double precision (its default is fp32).
    set_target(os.environ.get("Q2JEPA_CUDAQ_TARGET", "qpp-cpu"), os.environ.get("Q2JEPA_CUDAQ_PRECISION", "default"))
