import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta, Distribution, Normal, Uniform, kl_divergence
from torchtyping import TensorType

from dynamic_generation.experiments.train_base import BaseTrainer
from dynamic_generation.models.mlp import MLP
from dynamic_generation.types import Tensor


class BaseVAE(nn.Module):
    def __init__(self, trainer: BaseTrainer, prior: Distribution):
        super().__init__()
        self.trainer = trainer
        self.prior = prior

    def base_encode(self, x: Tensor) -> Distribution:
        raise NotImplementedError()

    def decode(self, z: TensorType["batch", "hidden"]) -> Distribution:
        raise NotImplementedError()

    def encode(self, x: Tensor):
        Z_hat = self.base_encode(x)

        if self.training:
            z_hat = Z_hat.rsample()
        else:
            z_hat = Z_hat.mean

        return {"z_hat": z_hat, "Z_hat": Z_hat}

    def generate(self, n: int, device):
        z = self.prior.sample((n,)).to(device=device)
        x = self.decode(z)
        return x.sample()

    def forward(self, x: Tensor):
        encoded = self.encode(x)
        X_hat = self.decode(encoded["z_hat"])
        return {"X_hat": X_hat} | encoded

    def loss(self, x, beta: float = 1):
        out = self.forward(x)
        X_hat = out["X_hat"]
        recon_loss = -X_hat.log_prob(x).mean()

        Z_hat = out["Z_hat"]
        prior = self.prior.expand(Z_hat.batch_shape)
        kl_loss = kl_divergence(Z_hat, prior).mean()

        loss = recon_loss + beta * kl_loss

        self.trainer.log("recon_loss", recon_loss.item(), "mean")
        self.trainer.log("kl_loss", kl_loss.item(), "mean")
        self.trainer.log("loss", loss.item(), "mean")

        return loss


class UniformBetaVAE(BaseVAE):
    def __init__(
        self,
        trainer: BaseTrainer,
        z_dim: int,
        hidden_dim: int,
        input_dim: int,
        n_layers: int,
    ):
        prior = Uniform(torch.zeros(z_dim), torch.ones(z_dim))
        super().__init__(trainer, prior)

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_layers = n_layers

        enc_dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [2 * z_dim]
        dec_dims = [z_dim] + [hidden_dim] * (n_layers - 1) + [input_dim]

        self.encoder = MLP(enc_dims, activation=nn.GELU())
        self.decoder = MLP(dec_dims, activation=nn.GELU())
        self.std = nn.parameter.Parameter(torch.tensor(1.0))

    def base_encode(self, x: Tensor) -> Distribution:
        h = self.encoder(x) ** 2
        a, b = h.chunk(2, dim=-1)
        return Beta(a, b)

    def decode(self, z: TensorType["batch", "hidden"]):
        mean = self.decoder(z)
        X_hat = Normal(loc=mean, scale=self.std)
        return X_hat
