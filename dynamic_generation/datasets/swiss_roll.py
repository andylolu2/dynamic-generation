from dataclasses import dataclass
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset

from dynamic_generation.datasets.base import BaseDataModule
from dynamic_generation.datasets.utils import infinite_loader


@dataclass
class SwissRollDataModule(BaseDataModule):
    std: float
    t_offset: float
    t_len: float

    def train_loader(self, batch_size: int, size: int) -> Iterator:
        assert size > 0
        ds = SwissRoll(size, self.std, self.t_offset, self.t_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        loader = infinite_loader(loader)
        return loader

    def eval_loader(self, batch_size: int, size: int) -> Iterable:
        assert size > 0
        ds = SwissRoll(size, self.std, self.t_offset, self.t_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader


class SwissRoll(Dataset):
    def __init__(
        self,
        size: int = 5000,
        std: float = 0.2,
        t_offset: float = 1.2 * np.pi,
        t_len: float = 2.6 * np.pi,
    ):
        self.size = size
        self.std = std
        self.t_offset = t_offset
        self.t_len = t_len
        self.data = self._build()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def _build(self):
        t = self.t_len * np.random.rand(self.size) + self.t_offset
        t = t[:, None]
        y = np.concatenate((t * np.cos(t), t * np.sin(t)), axis=-1)
        z = np.random.normal(scale=self.std, size=(self.size, 2))
        return y + z


if __name__ == "__main__":
    ds = SwissRoll(1000)

    plt.scatter(x=ds[:, 0], y=ds[:, 1], s=2)
    plt.gca().set_aspect("equal")

    plt.show()
