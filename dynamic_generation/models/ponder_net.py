"""Implements the PonderNet model.

Link to original paper: https://arxiv.org/abs/2107.05407

"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, kl_divergence
from torchtyping import TensorType

from dynamic_generation.experiments.train_base import BaseTrainer
from dynamic_generation.models.ponder_module import PonderModule
from dynamic_generation.types import Tensor
from dynamic_generation.utils.distributions import FiniteDiscrete, TruncatedGeometric


def geometric(p, N):
    ps = torch.full((1, N), p)
    dist = exclusive_cumprod(ps)
    dist /= dist.sum(-1, keepdim=True)
    return dist


def exclusive_cumprod(ps: Tensor, dim=-1):
    x = torch.cumprod(1 - ps, dim=dim)
    x = torch.roll(x, shifts=1, dims=dim)
    zero = torch.tensor(0, device=x.device)
    x = x.index_fill(dim=dim, index=zero, value=1)
    x *= ps
    return x


def calc_steps(epsilon: TensorType[()], lambda_p: TensorType[()]):
    return int(torch.ceil(torch.log(epsilon) / torch.log(1 - lambda_p)))


class PonderNet(nn.Module):
    def __init__(
        self,
        epsilon: float,
        lambda_p: float,
        beta: float,
        N_max: int,
        ponder_module: PonderModule,
        trainer: BaseTrainer,
    ):
        """
        Initializes a PonderNet module.

        Args:
            epsilon (float): The maximum probability to ignore when rolling out.
            lambda_p (float): The parameter to the prior geometric distribution.
            beta (float): The weighting of the prior term in the loss.
            N_max (int): The maximum unroll length.
            loss_fn ([Tensor, Tensor] -> TensorType["batch", "N"]): The target loss function.
            ponder_module (PonderModule): The ponder module that implements the dynamic compute.
            trainer (BaseTrainer): The trainer for logging.
        """
        super().__init__()
        self.epsilon = torch.tensor(epsilon)
        self.lambda_p = torch.tensor(lambda_p)
        self.beta = torch.tensor(beta)
        self.N_max = N_max
        self.ponder_module = ponder_module
        self.trainer = trainer

    def forward(self, x: TensorType["batch", "features"]):
        ys, halt_dist = self.ponder_module.eps_forward(x, self.epsilon, self.N_max)

        # log metrics
        self.trainer.log("halt_normalised_mean", halt_dist.mean.mean().item(), "mean")
        self.trainer.log("n_steps", halt_dist._num_events, "mean")

        return ys, halt_dist

    def prior(self, length: int):
        return TruncatedGeometric(self.lambda_p, length)

    def loss(
        self,
        y_true: TensorType["batch", "output"],
        y_pred: Distribution,
        halt_dist: FiniteDiscrete,
    ):
        L_rec = -y_pred.log_prob(y_true).mean()
        L_reg = kl_divergence(halt_dist, self.prior(halt_dist._num_events)).mean()
        loss = L_rec + self.beta * L_reg

        self.trainer.log("loss", loss.item(), "mean")
        self.trainer.log("loss_rec", L_rec.item(), "mean")
        self.trainer.log("loss_reg", L_reg.item(), "mean")

        return loss
