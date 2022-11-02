from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dynamic_generation.datasets.base import BaseDataModule
from dynamic_generation.datasets.utils import infinite_loader


@dataclass
class ParityDataModule(BaseDataModule):
    dim: int = 64

    def train_loader(self, batch_size: int, size: int = -1) -> Iterator:
        if size < 0:
            ds = InfiniteParity(self.dim)
            loader = DataLoader(ds, batch_size=batch_size)
            loader = iter(loader)
        else:
            ds = Parity(self.dim, size)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
            loader = infinite_loader(loader)
        return loader

    def eval_loader(self, batch_size: int, size: int) -> Iterable:
        assert size > 0
        ds = Parity(self.dim, size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader


def build_one(dim: int):
    # n ~ U(1, self.dim), where n is the number of 1 or -1
    thresh = torch.randint(1, dim + 1, size=())
    x = torch.arange(dim) < thresh

    # set x randomly to 1 or -1
    plus_or_minus_one = 2 * torch.randint(2, size=(dim,)) - 1
    x = x * plus_or_minus_one

    # shuffle x
    x = x[torch.randperm(dim)]

    # count x to generate the labels
    y = torch.abs(x).sum(dim=-1, keepdim=True)
    y = y % 2

    return x, y


class InfiniteParity(IterableDataset):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def __iter__(self):
        while True:
            yield {"data": build_one(self.dim, self.min_n, self.max_n)}


class Parity(Dataset):
    def __init__(self, dim: int = 64, size: int = 5000):
        super().__init__()
        self.dim = dim
        self.size = size
        self.xs, self.ys = self._build()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"data": (self.xs[idx], self.ys[idx])}

    def _build(self):
        xs, ys = [], []
        for _ in range(self.size):
            x, y = build_one(self.dim)
            xs.append(x)
            ys.append(y)

        xs = torch.stack(xs)
        ys = torch.stack(ys)

        return xs, ys


if __name__ == "__main__":
    ds = Parity(dim=8)
    print(ds[:10])

    ds = iter(InfiniteParity(dim=8))
    print(next(ds))
