import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

from dynamic_generation.experiments.train_base import BaseTrainer
from dynamic_generation.experiments.utils.metrics import global_metrics
from dynamic_generation.utils.stability import safe_log


class ToyGenerator(nn.Module):
    """This generator evaluates the log likelihood by brute-force"""

    def __init__(
        self,
        trainer: BaseTrainer,
        z_dim: int,
        hidden_dim: int,
        output_dim: int,
        z_samples: int,
        std: float,
    ):
        super().__init__()
        self.trainer = trainer
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

    def forward(self, z: TensorType["batch", "hidden"]):
        return self.generator(z)

    def generate(self, n: int, device):
        z = torch.rand(n, self.z_dim, device=device)
        return self.forward(z)

    def loss(self, x: TensorType["batch", "output"]):
        z = torch.rand(1, self.z_samples, self.z_dim, device=x.device)
        x_hat = self.forward(z)

        d = torch.linalg.norm(x.unsqueeze(1) - x_hat, dim=-1)
        lik_per_z = torch.exp(-0.5 * (d / self.std) ** 2) + 1e-10
        log_lik = torch.mean(safe_log(lik_per_z.mean(dim=1)))
        loss = -log_lik

        global_metrics.log("loss", loss.item(), "mean")
        return loss
