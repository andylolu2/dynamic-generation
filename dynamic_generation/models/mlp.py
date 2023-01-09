from typing import Sequence

from torch import nn

from dynamic_generation.types import Tensor

from .utils import load_activation


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], activation: str = "ReLU") -> None:
        super().__init__()
        self.dims = dims
        self.n_layers = len(dims) - 1
        self.activation = load_activation(activation)

        self.layers = nn.ModuleList()

        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                self.layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
                self.layers.append(nn.BatchNorm1d(dims[i + 1]))
                self.layers.append(self.activation)
            else:
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
