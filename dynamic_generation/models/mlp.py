from typing import Sequence

from torch import nn


class MLP(nn.Sequential):
    def __init__(self, dims: Sequence[int], activation: nn.Module):
        layers = []
        for dim1, dim2 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(dim1, dim2))
            layers.append(activation)
        layers.pop(-1)
        super().__init__(*layers)
