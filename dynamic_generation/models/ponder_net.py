"""Implements the PonderNet model.

Link to original paper: https://arxiv.org/abs/2107.05407

"""
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

from dynamic_generation.experiments.train_base import BaseTrainer
from dynamic_generation.types import Tensor


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


class PonderModule(nn.Module):
    def forward(
        self,
        x: TensorType["batch", "features"],
        N: int,
    ) -> tuple[TensorType["batch", "ponder", "outputs"], TensorType["batch", "N"]]:
        """Ponders for N steps"""
        raise NotImplementedError()

    def eps_forward(
        self,
        x: TensorType["batch", "features"],
        epsilon: TensorType[()],
        N_max: int = 20,
    ) -> tuple[TensorType["batch", "N", "outputs"], TensorType["batch", "N"]]:
        """Ponders until p_unhalted <= epsilon, or reached N_max steps"""
        raise NotImplementedError()


class RnnPonderModule(PonderModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.halt_logits = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: TensorType["batch", "features"],
        N: int,
    ) -> tuple[TensorType["batch", "N", "outputs"], TensorType["batch", "N"]]:
        xs = x.unsqueeze(1).expand((-1, N, -1))
        hs, _ = self.rnn(xs)
        ys = self.output(hs)
        halt_ns = self.halt_logits(hs).sigmoid().squeeze(-1)

        return ys, halt_ns

    def eps_forward(
        self,
        x: TensorType["batch", "features"],
        epsilon: TensorType[()],
        N_max: int = 20,
    ) -> tuple[TensorType["batch", "N", "outputs"], TensorType["batch", "N"]]:
        x = x.unsqueeze(1)

        h = None
        p_unhalted = torch.ones(x.shape[0], device=x.device)
        ys = []
        ps = []
        step = 0
        while (p_unhalted > epsilon).any() and step < N_max:
            out, h = self.rnn(x, h)
            p_halt = self.halt_logits(out).sigmoid().flatten()

            ys.append(self.output(out).squeeze(1))
            ps.append(p_unhalted * p_halt)
            p_unhalted = p_unhalted * (1 - p_halt)
            step += 1

        ys = torch.stack(ys).transpose(0, 1)  # batch first
        ps = torch.stack(ps).transpose(0, 1)  # batch first

        return ys, ps


class PonderNet(nn.Module):
    def __init__(
        self,
        epsilon: float,
        lambda_p: float,
        beta: float,
        N_max: int,
        loss_fn: Callable[[Tensor, Tensor], TensorType["batch", "N"]],
        ponder_module: PonderModule,
        trainer: BaseTrainer,
    ):
        super().__init__()
        self.epsilon = torch.tensor(epsilon)
        self.lambda_p = torch.tensor(lambda_p)
        self.beta = torch.tensor(beta)
        self.N_max = N_max
        self.loss_fn = loss_fn
        self.ponder_module = ponder_module
        self.trainer = trainer

    def forward(self, x: TensorType["batch", "features"]):
        ys, ps = self.ponder_module.eps_forward(x, self.epsilon, self.N_max)

        ps_norm = ps.sum(-1, keepdim=True)
        ps_normalized = ps / ps_norm

        # log metrics
        ps_cumsum = ps.cumsum(-1)
        ps_cumsum[..., -1] = 1  # use last position as "fallback"
        ps_median = (ps_cumsum > 0.5).int().argmax(-1).to(ps.dtype)
        self.trainer.log("ps_norm", ps_norm.mean().item(), "mean")
        self.trainer.log("ps_median", ps_median.mean().item(), "mean")
        self.trainer.log("n_steps", ys.shape[1], "mean")

        return ys, ps_normalized

    def prior(self, length: int):
        return geometric(self.lambda_p, length)

    def l_reg(self, ps: TensorType["batch", "N"]) -> TensorType["batch"]:
        prior = self.prior(ps.shape[1]).to(ps.device)
        return F.kl_div((ps + 1e-6).log(), prior, reduction="none").sum(-1)

    def loss(
        self,
        ys_target: TensorType["batch", "output"],
        ys: TensorType["batch", "N", "output"],
        ps: TensorType["batch", "N"],
    ):
        ys_target_unrolled = ys_target.unsqueeze(1)

        losses: TensorType["batch", "N"]
        losses = self.loss_fn(ys, ys_target_unrolled)

        weighted_loss: TensorType["batch"]
        weighted_loss = (ps * losses).sum(-1)

        L_rec = weighted_loss.mean()
        L_reg = self.l_reg(ps).mean()
        loss = L_rec + self.beta * L_reg

        self.trainer.log("loss", loss.item(), "mean")
        self.trainer.log("loss_rec", L_rec.item(), "mean")
        self.trainer.log("loss_reg", L_reg.item(), "mean")

        return loss
