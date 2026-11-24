import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "taxibj_baselines", Path(__file__).parents[1] / "scripts/taxibj_baselines.py"
)
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)


def test_copy_last_uses_the_last_observed_map():
    obs = torch.zeros(2, 3, 2, 4, 4, dtype=torch.float16)
    obs[:, -1] = 0.5
    fut = torch.full_like(obs, 0.75)
    assert tb.copy_last_mse(obs, fut) == 0.0625


def test_train_mean_keeps_the_horizon_step():
    train = torch.zeros(2, 3, 2, 4, 4, dtype=torch.float16)
    train[0, 1] = 1.0  # mean of step 1 is 0.5, steps 0 and 2 are 0
    fut = torch.zeros(5, 3, 2, 4, 4, dtype=torch.float16)
    fut[:, 1] = 0.5
    assert tb.train_mean_mse(train, fut) == 0.0
