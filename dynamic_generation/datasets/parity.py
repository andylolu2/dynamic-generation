from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dynamic_generation.datasets.base import BaseDataModule
from dynamic_generation.datasets.utils import infinite_loader


@dataclass
class ParityDataModule(BaseDataModule):
    dim: int = 64

    def train_loader(
        self,
        batch_size: int,
        size: int,
        min_n: int,
        max_n: int,
    ) -> Iterator:
        if size < 0:
            ds = InfiniteParity(self.dim, min_n, max_n)
            loader = DataLoader(ds, batch_size=batch_size)
            loader = iter(loader)
        else:
            ds = Parity(self.dim, size, min_n, max_n)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
            loader = infinite_loader(loader)
        return loader

    def eval_loader(
        self,
        batch_size: int,
        size: int,
        min_n: int,
        max_n: int,
    ) -> Iterable:
        assert size > 0
        ds = Parity(self.dim, size, min_n, max_n)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader


def build_one(dim: int, min_n: int, max_n: int):
    assert 0 <= min_n <= max_n <= dim

    # n ~ U(1, self.dim), where n is the number of 1 or -1
    thresh = torch.randint(min_n, max_n + 1, size=())
    x = torch.arange(dim) < thresh

    # set x randomly to 1 or -1
    plus_or_minus_one = 2 * torch.randint(2, size=(dim,)) - 1
    x = x * plus_or_minus_one

    # shuffle x
    x = x[torch.randperm(dim)]

    # count x to generate the labels
    y = torch.abs(x).sum(dim=-1)
    y = y % 2

    return x, y


class InfiniteParity(IterableDataset):
    def __init__(self, dim: int = 64, min_n: int = 1, max_n: int | None = None):
        super().__init__()
        self.dim = dim
        self.min_n = min_n
        self.max_n = dim if max_n is None else max_n

    def __iter__(self):
        while True:
            yield {"data": build_one(self.dim, self.min_n, self.max_n)}


class Parity(Dataset):
    def __init__(
        self,
        dim: int = 64,
        size: int = 5000,
        min_n: int = 1,
        max_n: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.size = size
        self.min_n = min_n
        self.max_n = dim if max_n is None else max_n
        self.xs, self.ys = self._build()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"data": (self.xs[idx], self.ys[idx])}

    def _build(self):
        xs, ys = [], []
        for _ in range(self.size):
            x, y = build_one(self.dim, self.min_n, self.max_n)
            xs.append(x)
            ys.append(y)

        xs = torch.stack(xs)
        ys = torch.stack(ys)

        return xs, ys


if __name__ == "__main__":
    ds = Parity(dim=8)
    print(ds[:10])

    ds = iter(InfiniteParity(dim=8, min_n=1, max_n=4))
    print(next(ds))
