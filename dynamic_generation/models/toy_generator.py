import torch
import torch.distributions as D
from torch import nn

from dynamic_generation.types import Tensor
from dynamic_generation.utils.metrics import global_metrics
from dynamic_generation.utils.stability import safe_log


class ToyGenerator(nn.Module):
    """This generator evaluates the log likelihood by brute-force"""

    def __init__(
        self, z_dim: int, hidden_dim: int, output_dim: int, z_samples: int, std: float
    ):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.z_samples = z_samples
        self.std = std
        self.generator = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: Tensor):
        mean = self.generator(z)
        X_hat = D.Independent(D.Normal(mean, self.std), 1)
        return X_hat

    def generate(self, n: int, device):
        z = torch.rand(n, self.z_dim, device=device)
        X_hat = self.forward(z)
        return X_hat.mean

    def loss(self, x: Tensor):
        z = torch.rand(1, self.z_samples, self.z_dim, device=x.device)
        X_hat = self.forward(z)
        X_hat = X_hat.expand((x.shape[0], self.z_samples))

        lik_per_z = X_hat.log_prob(x.unsqueeze(1)).exp()
        log_lik = torch.mean(safe_log(lik_per_z.mean(1)))
        loss = -log_lik

        global_metrics.log("loss", loss.item(), "mean")
        return loss
