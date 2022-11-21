import torch
import torch.distributions as D
from torch import nn
from torchtyping import TensorType

from dynamic_generation.experiments.train_base import BaseTrainer
from dynamic_generation.experiments.utils.metrics import global_metrics
from dynamic_generation.models.mlp import MLP
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
        Z_hat = self.base_encode(x)

        if self.training:
            z_hat = Z_hat.rsample()
        else:
            z_hat = Z_hat.mean

        return {"z_hat": z_hat, "Z_hat": Z_hat}

    def generate(self, n: int, device):
        z = self.prior.sample((n,)).to(device=device)  # type: ignore
        x, aux = self.decode(z)
        return x.sample(), aux

    def forward(self, x: Tensor):
        encoded = self.encode(x)
        X_hat, aux = self.decode(encoded["z_hat"])
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
        self,
        z_dim: int,
        hidden_dim: int,
        input_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers

        enc_dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [2 * z_dim]
        dec_dims = [z_dim] + [hidden_dim] * (n_layers - 1) + [input_dim]

        self.encoder = MLP(enc_dims, activation=nn.GELU())
        self.decoder = MLP(dec_dims, activation=nn.GELU())
        self.std = nn.parameter.Parameter(torch.tensor(1.0))

        self.register_buffer("prior_low", torch.zeros(z_dim))
        self.register_buffer("prior_high", torch.ones(z_dim))

    @property
    def prior(self):
        return D.Uniform(self.prior_low, self.prior_high)

    def base_encode(self, x: Tensor) -> D.Distribution:
        h = self.encoder(x) ** 2
        a, b = h.chunk(2, dim=-1)
        return D.Beta(a, b)

    def decode(self, z: TensorType["batch", "hidden"]):
        mean = self.decoder(z)
        X_hat = D.Normal(loc=mean, scale=self.std)
        return X_hat, {}


class DynamicVae(BaseVAE):
    def __init__(
        self,
        z_dim: int,
        enc_hidden_dim: int,
        enc_n_layers: int,
        dec_hidden_dim: int,
        dec_n_layers: int,
        input_dim: int,
        epsilon: float,
        lambda_p: float,
        beta: float,
        N_max: int,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.enc_n_layers = enc_n_layers
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_n_layers = dec_n_layers
        self.input_dim = input_dim

        enc_dims = [input_dim] + [enc_hidden_dim] * (enc_n_layers - 1) + [2 * z_dim]
        self.encoder = MLP(enc_dims, activation=nn.GELU())

        ponder_module = RNNPonderModule(z_dim, dec_hidden_dim, input_dim, dec_n_layers)
        self.decoder = PonderNet(
            epsilon=epsilon,
            lambda_p=lambda_p,
            beta=beta,
            N_max=N_max,
            ponder_module=ponder_module,
        )
        # self.std: Tensor
        # self.register_buffer("std", torch.tensor(0.001))
        self.std = nn.parameter.Parameter(torch.tensor(1.0))

        self.prior_low: Tensor
        self.prior_high: Tensor
        self.register_buffer("prior_low", torch.zeros(z_dim))
        self.register_buffer("prior_high", torch.ones(z_dim))

    @property
    def prior(self):
        return D.Uniform(self.prior_low, self.prior_high)

    def base_encode(self, x: Tensor) -> D.Distribution:
        h = self.encoder(x).abs()
        a, b = h.chunk(2, dim=-1)
        return D.Beta(a, b)

    def decode(self, z: TensorType["batch", "hidden"]):
        means, halt_dist = self.decoder.forward(z)
        cov = self.std**2 * torch.eye(self.input_dim, device=means.device)
        X_hats = D.MultivariateNormal(loc=means.sigmoid(), covariance_matrix=cov)
        halt_dist = halt_dist.expand(X_hats.batch_shape[:-1])
        X_hat = CustomMixture(halt_dist, X_hats)
        return X_hat, {"halt_dist": halt_dist}

    def generate(self, n: int, device):
        z = self.prior.sample((n,)).to(device=device)  # type: ignore
        x, aux = self.decode(z)
        sample, mix_sample = x.sample_detailed()
        return sample, aux | {"mix_sample": mix_sample}

    def loss(self, x: Tensor, out: dict, beta: float = 1):
        loss = super().loss(x, out, beta)
        ponder_loss = self.decoder.regularisation(out["halt_dist"], self.decoder.beta)

        global_metrics.log("total_loss", loss.item(), "mean")

        return loss + ponder_loss
