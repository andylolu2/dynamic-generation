import torch
from torch import nn
from torchtyping import TensorType

from dynamic_generation.types import Tensor
from dynamic_generation.utils.distributions import FiniteDiscrete
from dynamic_generation.utils.stability import safe_log


class PonderModule(nn.Module):
    def forward(
        self,
        x: TensorType["batch", "features"],
        N: int,
    ) -> tuple[TensorType["batch", "ponder", "outputs"], FiniteDiscrete]:
        """Ponders for N steps"""
        raise NotImplementedError()

    def eps_forward(
        self,
        x: TensorType["batch", "features"],
        epsilon: TensorType[()],
        N_max: int = 20,
    ) -> tuple[TensorType["batch", "N", "outputs"], FiniteDiscrete]:
        """Ponders until p_unhalted <= epsilon, or reached N_max steps"""
        raise NotImplementedError()


class RecurrentPonderModule(PonderModule):
    def recur(self, x: Tensor, h: Tensor | None, steps: int) -> tuple[Tensor, Tensor]:
        """
        Computes the final hidden state and digest of the intermediate hidden states.

        Args:
            x (Tensor): the inputs of shape `batch x dim`
            h (Tensor): the previous hidden state

        Returns:
            (Tensor, Tensor): Tuple of (next hidden state, hidden state digest).
                Next hidden state is meant to be used when calling this function
                the next time. The digest is meant to used for predictions. Digest
                will have shape `batch x steps x h_dim`.
        """
        raise NotImplementedError()

    def output(self, digest: Tensor) -> Tensor:
        """
        Computes the output given the hidden state digest.

        Args:
            digest (Tensor): The hidden state digest.

        Returns:
            Tensor: The target output along the pondering steps.
        """
        raise NotImplementedError()

    def halt_logits(self, digest: Tensor) -> Tensor:
        """
        Computes the halting logits given the hidden state digest.

        Args:
            digest (Tensor): The hidden state digest.

        Returns:
            Tensor: Tensor of the halting logits of shape `batch x 1`.
        """
        ...

    def forward(
        self,
        x: TensorType["batch", "features"],
        N: int,
    ) -> tuple[TensorType["batch", "N", "outputs"], FiniteDiscrete]:
        _, digest = self.recur(x, h=None, steps=N)
        ys = self.output(digest)  # batch x ponder x dim
        halt_logits = self.halt_logits(digest).squeeze(-1)  # batch x ponder
        halt_dist = FiniteDiscrete(logits=halt_logits)
        return ys, halt_dist

    def eps_forward(
        self,
        x: TensorType["batch", "features"],
        epsilon: TensorType[()],
        N_max: int = 20,
    ) -> tuple[TensorType["batch", "N", "outputs"], FiniteDiscrete]:
        h = None
        unhalted_logits = torch.zeros(x.shape[0], device=x.device)
        ys = []
        halt_logits = []
        step = 0
        while (unhalted_logits > epsilon.log()).any() and step < N_max:
            h, digest = self.recur(x, h, steps=1)
            p_halt = self.halt_logits(digest).sigmoid().flatten()

            ys.append(self.output(digest).squeeze(1))
            halt_logits.append(unhalted_logits + safe_log(p_halt))

            unhalted_logits += safe_log(1 - p_halt)
            step += 1

        ys = torch.stack(ys).permute(1, 0, 2)  # batch x dim x ponder
        halt_logits = torch.stack(halt_logits).permute(1, 0)  # batch x ponder
        halt_dist = FiniteDiscrete(logits=halt_logits)

        return ys, halt_dist


class RNNPonderModule(RecurrentPonderModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
    ):
        super().__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.ys = nn.Linear(hidden_size, output_size)
        self.h_logits = nn.Linear(hidden_size, 1)

    def recur(self, x: Tensor, h: Tensor | None, steps: int) -> tuple[Tensor, Tensor]:
        x = x.unsqueeze(1).expand((-1, steps, -1))
        digest, h_ = self.rnn(x, h)
        return h_, digest

    def output(self, digest: Tensor) -> Tensor:
        return self.ys(digest)

    def halt_logits(self, digest: Tensor) -> Tensor:
        return self.h_logits(digest)


class GRUPonderModule(RecurrentPonderModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
    ):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.ys = nn.Linear(hidden_size, output_size)
        self.h_logits = nn.Linear(hidden_size, 1)

    def recur(self, x: Tensor, h: Tensor | None, steps: int) -> tuple[Tensor, Tensor]:
        x = x.unsqueeze(1).expand((-1, steps, -1))
        digest, h_ = self.gru(x, h)
        return h_, digest

    def output(self, digest: Tensor) -> Tensor:
        return self.ys(digest)

    def halt_logits(self, digest: Tensor) -> Tensor:
        return self.h_logits(digest)
