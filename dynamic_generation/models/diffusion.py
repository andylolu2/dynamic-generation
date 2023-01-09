import math
from typing import Literal, overload

import torch
from torch import nn

from dynamic_generation.types import Shape, Tensor
from dynamic_generation.utils.metrics import global_metrics


class NoiseSchedule(nn.Module):
    def __init__(self, beta, alpha, alpha_bar):
        super().__init__()
        self.steps = len(beta)

        # For typing
        self.beta: Tensor
        self.alpha: Tensor
        self.alpha_bar: Tensor

        # Move to device
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)


class LinearNoiseSchedule(NoiseSchedule):
    def __init__(self, beta_1: float, beta_T: float, steps: int, scale: float):
        total_steps = int(steps * scale)
        beta = torch.linspace(beta_1, beta_T, total_steps) / scale
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, 0)
        super().__init__(beta, alpha, alpha_bar)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """Following original paper's implementation"""
        half = self.dim // 2
        i = torch.arange(half, device=t.device)
        freqs = torch.exp(-math.log(self.max_period) / (half - 1) * i)
        args = t.view(-1, 1).float() * freqs.view(1, -1)
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        return embedding


class Diffuser(nn.Module):
    def __init__(self, input_dims: Shape, noise_schedule_kwargs: dict):
        super().__init__()
        self.input_dims = input_dims
        self.ns = LinearNoiseSchedule(**noise_schedule_kwargs)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Predicts the noise added at time t which created the noisy input x_t"""
        raise NotImplementedError()

    def loss(self, x_0: Tensor) -> Tensor:
        """Loss calculated by the simplified loss objective."""
        x_t, t, eps = self.add_noise(x_0)
        eps_pred = self.forward(x_t, t)
        loss = ((eps - eps_pred) ** 2).mean()

        global_metrics.log("loss", loss.item(), "mean")
        return loss

    def add_noise(self, x_0: Tensor):  # x_0: (B,C,H,W)
        t = torch.randint(0, self.ns.steps, (x_0.shape[0],), device=x_0.device)  # (B,)
        x_t, eps = self.sample_q(x_0, t)
        return x_t, t, eps

    @overload
    def remove_noise(self, x_t: Tensor, t: Tensor, t_next=None, mode="ddpm") -> Tensor:
        ...

    @overload
    def remove_noise(
        self, x_t: Tensor, t: Tensor, t_next: Tensor, mode="ddim"
    ) -> Tensor:
        ...

    def remove_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        t_next: Tensor | None = None,
        mode="ddpm",
    ):
        factor = (1 - self.ns.alpha_bar) ** 0.5
        eps_pred = self.forward(x_t, t)

        t = t.view(-1, 1, 1, 1)  # needed for automatic broadcasting

        if mode == "ddpm":
            """See algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""
            x = (1 / self.ns.alpha[t] ** 0.5) * (
                x_t - ((1 - self.ns.alpha[t]) / factor[t]) * eps_pred
            )
            sigma_t = self.ns.beta[t] ** 0.5
            z = (t > 0) * torch.randn_like(x_t)
            x += sigma_t * z
        elif mode == "ddim":
            """See section 4.1 and C.1 in https://arxiv.org/pdf/2010.02502.pdf

            Note: alpha in the DDIM paper is actually alpha_bar in DDPM paper
            """
            assert t_next is not None
            t_next = t_next.view(-1, 1, 1, 1)

            x_0 = (x_t - factor[t] * eps_pred) / self.ns.alpha_bar[t] ** 0.5
            x_t_direction = factor[t_next] * eps_pred
            x = self.ns.alpha_bar[t_next] ** 0.5 * x_0 + x_t_direction
        else:
            raise ValueError(f"Invalid mode: {mode}")

        x = x.clip(-1, 1)

        return x

    @overload
    def generate(self, x_T: Tensor, *, mode: Literal["ddpm"] = ...) -> Tensor:
        ...

    @overload
    def generate(self, x_T: Tensor, steps: int, *, mode: Literal["ddim"]) -> Tensor:
        ...

    def generate(
        self,
        x_T: Tensor,
        steps: int | None = None,
        *,
        mode: Literal["ddpm", "ddim"] = "ddpm",
    ):
        x = x_T
        dummy = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)

        if mode == "ddpm":
            for t in range(self.ns.steps - 1, -1, -1):
                x = self.remove_noise(x, torch.full_like(dummy, t), mode=mode)
        elif mode == "ddim":
            assert steps is not None
            ts = self.time_steps(steps).tolist()
            for t, t_next in zip(ts[:-1], ts[1:]):
                x = self.remove_noise(
                    x,
                    torch.full_like(dummy, t),
                    torch.full_like(dummy, t_next),
                    mode=mode,
                )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return x

    def sample_q(self, x_0: Tensor, t: Tensor):
        alpha_t_bar = self.ns.alpha_bar[t]  # (B,)
        alpha_t_bar = alpha_t_bar.reshape(-1, 1, 1, 1)  # (B,1,1,1)

        eps = torch.randn_like(x_0)  # (B,C,H,W)
        x_t = (alpha_t_bar**0.5) * x_0 + ((1 - alpha_t_bar) ** 0.5) * eps  # (B,C,H,W)

        return x_t, eps

    def time_steps(self, steps: int):
        time_steps = torch.linspace(0, self.ns.steps - 1, steps + 1)
        time_steps = torch.round(time_steps).int()
        return time_steps.flip(0)
