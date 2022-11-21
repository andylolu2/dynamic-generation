import torch
import torch.distributions as D

from dynamic_generation.types import Tensor


class FiniteDiscrete(D.Categorical):
    """Implements a integer distribution with support [0, N)."""

    @property
    def N(self) -> int:
        return self._num_events

    @property
    def mean(self):
        return torch.sum(
            self.probs * torch.arange(self._num_events, device=self.probs.device),
            dim=-1,
        )

    @property
    def variance(self):
        E_x_squared = torch.sum(
            self.probs * torch.arange(self._num_events, device=self.probs.device) ** 2,
            dim=-1,
        )
        return E_x_squared - self.mean**2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(D.Categorical, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        new._num_events = self._num_events
        super(D.Categorical, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def unsqueeze(self, dim):
        if "probs" in self.__dict__:
            return FiniteDiscrete(probs=self.probs.unsqueeze(dim=dim))
        if "logits" in self.__dict__:
            return FiniteDiscrete(logits=self.logits.unsqueeze(dim=dim))


class TruncatedGeometric(FiniteDiscrete):
    def __init__(self, p: Tensor, N: int, validate_args=None):
        logits = p.log() + torch.arange(N, device=p.device) * (1 - p).log()
        super().__init__(logits=logits, validate_args=validate_args)


class CustomMixture(D.MixtureSameFamily):
    def sample_detailed(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1))
            )
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
            )

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim), mix_sample_r.squeeze(gather_dim)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dist = TruncatedGeometric(0.1, 10)
    x = dist.sample((5000,))

    dist = FiniteDiscrete(probs=torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.2, 0.2]]))
    print(dist.mean)
    print(dist.variance)

    plt.hist(x[:], bins=10)
    plt.show()
