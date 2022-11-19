import torch
from torch.distributions import Categorical

from dynamic_generation.types import Tensor


class FiniteDiscrete(Categorical):
    """Implements a integer distribution with support [0, N)."""

    @property
    def mean(self):
        return torch.sum(self.probs * torch.arange(self._num_events), dim=-1)

    @property
    def variance(self):
        E_x_squared = torch.sum(
            self.probs * torch.arange(self._num_events) ** 2, dim=-1
        )
        return E_x_squared - self.mean**2


class TruncatedGeometric(FiniteDiscrete):
    def __init__(self, p: float | Tensor, N: int, validate_args=None):
        p_ = torch.tensor(p)
        logits = p_.log() + torch.arange(N) * (1 - p_).log()
        super().__init__(logits=logits, validate_args=validate_args)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dist = TruncatedGeometric(0.1, 10)
    x = dist.sample((5000,))

    dist = FiniteDiscrete(probs=torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.2, 0.2]]))
    print(dist.mean)
    print(dist.variance)

    plt.hist(x[:], bins=10)
    plt.show()
