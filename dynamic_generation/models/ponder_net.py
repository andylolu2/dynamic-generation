"""Implements the PonderNet model.

Link to original paper: https://arxiv.org/abs/2107.05407

"""
import torch
import torch.distributions as D
from torch import nn
from torchtyping import TensorType

from dynamic_generation.experiments.utils.metrics import global_metrics
from dynamic_generation.models.ponder_module import PonderModule
from dynamic_generation.types import Tensor
from dynamic_generation.utils.distributions import FiniteDiscrete, TruncatedGeometric


class PonderNet(nn.Module):
    def __init__(
        self,
        epsilon: float,
        lambda_p: float,
        beta: float,
        N_max: int,
        ponder_module: PonderModule,
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
        """
        super().__init__()
        self.N_max = N_max
        self.ponder_module = ponder_module

        # register buffers
        self.beta: Tensor
        self.epsilon: Tensor
        self.lambda_p: Tensor
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.register_buffer("lambda_p", torch.tensor(lambda_p))

    def forward(self, x: TensorType["batch", "features"], dynamic=False):
        if dynamic:
            ys, halt_dist = self.ponder_module.eps_forward(x, self.epsilon, self.N_max)
        else:
            ys, halt_dist = self.ponder_module.forward(x, self.N_max)

        # log metrics
        global_metrics.log(
            "halt_normalised_mean", halt_dist.mean.mean().item() + 1, "mean"
        )
        global_metrics.log("n_steps", halt_dist._num_events, "mean")

        return ys, halt_dist

    def prior(self, length: int):
        return TruncatedGeometric(self.lambda_p, length)

    def regularisation(
        self, halt_dist: FiniteDiscrete, beta: Tensor | int = 1, average=False
    ):
        if average:
            average_halt_dist = halt_dist.average()
            ponder_loss = D.kl_divergence(
                average_halt_dist, self.prior(average_halt_dist.N)
            )
        else:
            ponder_loss = D.kl_divergence(halt_dist, self.prior(halt_dist.N)).mean()
        ponder_loss *= beta
        global_metrics.log("ponder_loss", ponder_loss.item(), "mean")
        return ponder_loss

    def loss(
        self,
        y_true: TensorType["batch", "output"],
        y_pred: D.Distribution,
        halt_dist: FiniteDiscrete,
    ):
        target_loss = -y_pred.log_prob(y_true).mean()
        ponder_loss = self.regularisation(halt_dist, self.beta)
        loss = target_loss + ponder_loss

        global_metrics.log("target_loss", target_loss.item(), "mean")
        global_metrics.log("loss", loss.item(), "mean")

        return loss
