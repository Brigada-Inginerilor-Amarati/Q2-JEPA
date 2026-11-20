"""The paper's HEP18 encoder on structured states (App. A.2): a sanity baseline, not an image encoder.

Six scalars per agent (three (s / 50, v / 10) pairs) -> 64 -> 64 -> d, SiLU. Only HEP18 supports it.
"""

from torch import nn


class StateEncoder(nn.Module):
    def __init__(self, d: int = 16, inputs: int = 6, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, d))  # fmt: skip

    def forward(self, x):
        return self.net(x)
