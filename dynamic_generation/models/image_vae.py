from typing import Sequence

import torch
import torch.distributions as D
from torch import nn

from dynamic_generation.types import Tensor

from .blocks import BlockSequential, LinearBlock
from .vae import BaseVAE


class ImageVae(BaseVAE):
    def __init__(
        self,
        input_dims: tuple[int, int, int],
        z_dim: int,
        enc_dims: Sequence[int],
        dec_dims: Sequence[int],
    ):
        super().__init__()
        self.input_dims = input_dims
        self.z_dim = z_dim
        self.enc_dims = list(enc_dims)
        self.dec_dims = list(dec_dims)

        c, h, w = input_dims

        # encoder
        enc_dims = [c * h * w] + self.enc_dims + [z_dim * 2]
        self.encoder = BlockSequential(
            dims=enc_dims,
            first_block=LinearBlock(pre_block=nn.Flatten(), post_block=nn.ReLU()),
            block=LinearBlock(post_block=nn.ReLU()),
            last_block=LinearBlock(),
        )

        # decoder
        dec_dims = [z_dim] + self.dec_dims + [c * h * w]
        self.decoder = BlockSequential(
            dims=dec_dims,
            block=LinearBlock(post_block=nn.ReLU()),
            last_block=LinearBlock(post_block=nn.Unflatten(-1, (c, h, w))),
        )

        self.prior_mean: Tensor
        self.prior_std: Tensor
        self.register_buffer("prior_mean", torch.zeros(z_dim))
        self.register_buffer("prior_std", torch.ones(z_dim))

    @property
    def prior(self):
        return D.Independent(D.Normal(self.prior_mean, self.prior_std), 1)

    def base_encode(self, x: Tensor):
        # x: N,C,H,W
        z = self.encoder(x)
        z_mean, z_log_var = z.chunk(2, dim=-1)
        return D.Independent(D.Normal(z_mean, z_log_var.exp()), 1)

    def decode(self, z: Tensor):
        x = self.decoder(z)
        X_hat = D.Independent(D.ContinuousBernoulli(logits=x), 3)
        return X_hat, {}

    def generate(self, n: int, device):
        z = self.prior.sample((n,)).to(device=device)  # type: ignore
        x, aux = self.decode(z)
        return x.base_dist.probs, aux | {"z": z}
