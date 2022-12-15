from typing import Sequence

import torch
import torch.distributions as D
from torch import nn
from torchtyping import TensorType

from dynamic_generation.experiments.utils.metrics import global_metrics
from dynamic_generation.models.block_sequential import BlockSequential
from dynamic_generation.models.blocks import LinearBlock
from dynamic_generation.models.ponder_module import GRUPonderModule, RNNPonderModule
from dynamic_generation.models.ponder_net import PonderNet
from dynamic_generation.types import Tensor
from dynamic_generation.utils.distributions import CustomMixture


class BaseVAE(nn.Module):
    @property
    def prior(self) -> D.Distribution:
        raise NotImplementedError()

    def base_encode(self, x: Tensor) -> D.Distribution:
        raise NotImplementedError()

    def decode(self, z: TensorType["batch", "hidden"]) -> tuple[D.Distribution, dict]:
        raise NotImplementedError()

    def encode(self, x: Tensor):
        return {"Z_hat": self.base_encode(x)}

    def generate(self, n: int, device):
        z = self.prior.sample((n,)).to(device=device)  # type: ignore
        x, aux = self.decode(z)
        return x.mean, aux

    def forward(self, x: Tensor):
        encoded = self.encode(x)
        X_hat, aux = self.decode(encoded["Z_hat"].rsample())
        return {"X_hat": X_hat} | encoded | aux

    def loss(self, x: Tensor, out: dict, beta: float = 1):
        """
        Computes the loss.

        Args:
            x (Tensor): The input to auto-encode.
            out (dict): The output from self.forward
            beta (float, optional): The weighting in beta-vae. Defaults to 1.

        Returns:
            Tensor: The loss.
        """
        X_hat: D.Distribution = out["X_hat"]
        Z_hat: D.Distribution = out["Z_hat"]
        recon_loss = -X_hat.log_prob(x).mean()

        prior = self.prior.expand(Z_hat.batch_shape)
        kl_loss = D.kl_divergence(Z_hat, prior).mean()

        loss = recon_loss + beta * kl_loss

        global_metrics.log("recon_loss", recon_loss.item(), "mean")
        global_metrics.log("kl_loss", kl_loss.item(), "mean")
        global_metrics.log("vae_loss", loss.item(), "mean")

        return loss


class UniformBetaVAE(BaseVAE):
    def __init__(
        self, z_dim: int, hidden_dim: int, input_dim: int, n_layers: int, std: float
    ):
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers

        enc_dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [2 * z_dim]
        dec_dims = [z_dim] + [hidden_dim] * (n_layers - 1) + [input_dim]

        def block_fn(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())

        def last_block_fn(in_dim, out_dim):
            return nn.Linear(in_dim, out_dim)

        self.encoder = BlockSequential(
            enc_dims, block=block_fn, last_block=last_block_fn
        )
        self.decoder = BlockSequential(
            dec_dims, block=block_fn, last_block=last_block_fn
        )

        self.std: Tensor
        self.register_buffer("std", torch.tensor(std))

        self.prior_low: Tensor
        self.prior_high: Tensor
        self.register_buffer("prior_low", torch.zeros(z_dim))
        self.register_buffer("prior_high", torch.ones(z_dim))

    @property
    def prior(self):
        return D.Independent(D.Uniform(self.prior_low, self.prior_high), 1)

    def base_encode(self, x: Tensor):
        h = self.encoder(x).sigmoid() * 400
        a, b = h.chunk(2, dim=-1)
        return D.Independent(D.Beta(a, b), 1)

    def decode(self, z: TensorType["batch", "hidden"]):
        mean = self.decoder(z)
        X_hat = D.Independent(D.Normal(mean, self.std), 1)
        return X_hat, {}


class DynamicVae(BaseVAE):
    def __init__(
        self,
        z_dim: int,
        enc_dims: Sequence[int],
        dec_hidden_dim: int,
        dec_n_layers: int,
        input_dim: int,
        epsilon: float,
        lambda_p: float,
        beta: float,
        N_max: int,
        average_halt_dist: bool,
        std: float,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.enc_dims = list(enc_dims)
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_n_layers = dec_n_layers
        self.input_dim = input_dim

        enc_dims = [input_dim] + self.enc_dims + [2 * z_dim]
        self.encoder = BlockSequential(
            enc_dims,
            block=LinearBlock(post_block=nn.GELU()),
            last_block=LinearBlock(),
        )

        # ponder_module = GRUPonderModule(z_dim, dec_hidden_dim, input_dim, dec_n_layers)
        ponder_module = RNNPonderModule(z_dim, dec_hidden_dim, input_dim, dec_n_layers)
        self.decoder = PonderNet(
            epsilon=epsilon,
            lambda_p=lambda_p,
            beta=beta,
            N_max=N_max,
            ponder_module=ponder_module,
            average_halt_dist=average_halt_dist,
        )

        self.std: Tensor
        self.register_buffer("std", torch.tensor(std))

        self.prior_low: Tensor
        self.prior_high: Tensor
        self.register_buffer("prior_low", torch.zeros(z_dim))
        self.register_buffer("prior_high", torch.ones(z_dim))

    @property
    def prior(self):
        return D.Independent(D.Uniform(self.prior_low, self.prior_high), 1)

    def base_encode(self, x: Tensor):
        h = self.encoder(x).sigmoid() * 400
        a, b = h.chunk(2, dim=-1)
        return D.Independent(D.Beta(a, b), 1)

    def decode(self, z: TensorType["batch", "hidden"]):
        means, halt_dist = self.decoder.forward(z)
        X_hats = D.Independent(D.Normal(means, self.std), 1)
        halt_dist = halt_dist.expand(X_hats.batch_shape[:-1])
        X_hat = CustomMixture(halt_dist, X_hats)
        return X_hat, {"halt_dist": halt_dist}

    def generate(self, n: int, device):
        z = self.prior.sample((n,)).to(device=device)  # type: ignore
        x, aux = self.decode(z)
        sample, mix_sample = x.sample_detailed(method="mean")
        return sample, aux | {"mix_sample": mix_sample}

    def loss(self, x: Tensor, out: dict, beta: float = 1):
        loss = super().loss(x, out, beta)
        ponder_loss = self.decoder.regularisation(out["halt_dist"], self.decoder.beta)

        global_metrics.log("total_loss", loss.item(), "mean")

        return loss + ponder_loss
