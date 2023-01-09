from typing import Sequence

import torch
from torch import nn

from dynamic_generation.types import Tensor

from .diffusion import Diffuser, TimeEmbedding
from .mlp import MLP


class MLPDiffuser(Diffuser):
    def __init__(
        self, dims: Sequence[int], time_embedding_kwargs: dict, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dims = list(dims)

        total = torch.Size(self.input_dims).numel()

        self.time_embed = TimeEmbedding(**time_embedding_kwargs)
        self.body = MLP([total + self.time_embed.dim, *self.dims, total])
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(-1, self.input_dims)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        x = self.flatten(x_t)
        t = self.time_embed(t)
        x = torch.concat((x, t), dim=-1)
        x = self.body(x)
        x = self.unflatten(x)
        return x
