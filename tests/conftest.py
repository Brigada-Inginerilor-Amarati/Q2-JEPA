import os

import pytest

from q2jepa.quantum import set_target


@pytest.fixture(scope="session", autouse=True)
def _cudaq_target():
    set_target(os.environ.get("Q2JEPA_CUDAQ_TARGET", "qpp-cpu"))
