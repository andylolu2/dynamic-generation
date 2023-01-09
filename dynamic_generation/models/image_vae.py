from typing import Sequence

import torch
import torch.distributions as D
from torch import nn

from dynamic_generation.types import Shape, Tensor
from dynamic_generation.utils.distributions import CustomMixture
from dynamic_generation.utils.metrics import global_metrics

from .mlp import MLP
from .ponder_module import RNNPonderModule
from .ponder_net import PonderNet
from .vae import BaseVAE


class ImageVae(BaseVAE):
    def __init__(
        self,
        input_dims: Shape,
        z_dim: int,
        enc_dims: Sequence[int],
        dec_dims: Sequence[int],
    ):
        super().__init__()
        self.input_dims = input_dims
        self.z_dim = z_dim
        self.enc_dims = list(enc_dims)
        self.dec_dims = list(dec_dims)

        total = torch.Size(input_dims).numel()

        # encoder
        self.flatten = nn.Flatten()
        self.encoder = MLP([total, *self.enc_dims, z_dim * 2])

        # decoder
        self.decoder = MLP([z_dim, *self.dec_dims, total])
        self.unflatten = nn.Unflatten(-1, input_dims)

        self.prior_mean: Tensor
        self.prior_std: Tensor
        self.register_buffer("prior_mean", torch.zeros(z_dim))
        self.register_buffer("prior_std", torch.ones(z_dim))

    @property
    def prior(self):
        return D.Independent(D.Normal(self.prior_mean, self.prior_std), 1)

    def base_encode(self, x: Tensor):
        # x: N,C,H,W
        x = self.flatten(x)
        z = self.encoder(x)
        z_mean, z_log_var = z.chunk(2, dim=-1)
        return D.Independent(D.Normal(z_mean, z_log_var.exp()), 1)

    def decode(self, z: Tensor):
        x = self.decoder(z)
        x = self.unflatten(x)
        X_hat = D.Independent(D.ContinuousBernoulli(logits=x), 3)
        return X_hat, {}

    def generate(self, n: int, device):
        z = self.prior.sample((n,)).to(device=device)  # type: ignore
        x, aux = self.decode(z)
        return x.base_dist.probs, aux | {"z": z}


class DynamicImageVae(BaseVAE):
    def __init__(
        self,
        z_dim: int,
        input_dims: tuple[int, int, int],
        enc_dims: Sequence[int],
        ponder_module_kwargs,
        ponder_net_kwargs,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.z_dim = z_dim
        self.enc_dims = list(enc_dims)

        c, h, w = input_dims

        enc_dims = [c * h * w] + self.enc_dims + [2 * z_dim]
        self.encoder = nn.Sequential(nn.Flatten(), MLP(enc_dims))

        self.decoder = PonderNet(
            ponder_module=RNNPonderModule(
                input_size=z_dim, output_size=c * h * w, **ponder_module_kwargs
            ),
            **ponder_net_kwargs
        )
        self.unflatten = nn.Unflatten(-1, (c, h, w))

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
        x_hats, halt_dist = self.decoder(z)
        x_hats = self.unflatten(x_hats)
        X_hats = D.Independent(D.ContinuousBernoulli(logits=x_hats), 3)
        X_hat = CustomMixture(halt_dist, X_hats)
        return X_hat, {"halt_dist": halt_dist, "X_hats": X_hats}

    def generate(self, n: int, device):
        z = self.prior.sample((n,))  # type: ignore
        x, aux = self.decode(z)
        sample, mix_sample = x.sample_detailed(mode="deterministic")
        return sample, aux | {"z": z, "mix_sample": mix_sample}

    def loss(self, x: Tensor, out: dict, beta: float = 1):
        loss = super().loss(x, out, beta)
        ponder_loss = self.decoder.regularisation(out["halt_dist"], self.decoder.beta)

        global_metrics.log("total_loss", loss.item(), "mean")

        return loss + ponder_loss
