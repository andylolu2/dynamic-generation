from dataclasses import dataclass
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset

from dynamic_generation.datasets.base import BaseDataModule
from dynamic_generation.datasets.utils import infinite_loader


@dataclass
class LollipopDataModule(BaseDataModule):
    std: float
    arm_len: float
    swirl_freq: float
    n_rotations: int
    swirl_prop: float

    def dataset(self, size: int):
        return Lollipop(
            size,
            self.std,
            self.arm_len,
            self.swirl_freq,
            self.n_rotations,
            self.swirl_prop,
        )

    def train_loader(self, batch_size: int, size: int) -> Iterator:
        assert size > 0
        ds = self.dataset(size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        loader = infinite_loader(loader)
        return loader

    def eval_loader(self, batch_size: int, size: int) -> Iterable:
        assert size > 0
        ds = self.dataset(size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader


class Lollipop(Dataset):
    def __init__(
        self,
        size: int = 5000,
        std: float = 0.2,
        arm_len: float = 15,
        swirl_freq: float = 1.5,
        n_rotations: int = 2,
        swirl_prop: float = 0.8,
    ):
        self.size = size
        self.std = std
        self.swirl_prop = swirl_prop

        # arm
        self.arm_lim = -np.sqrt(arm_len**2 / 2)

        # swirl
        self.swirl_freq = swirl_freq
        self.t_final = (n_rotations + 5 / 8) / self.swirl_freq * 2 * np.pi
        self.x_offset = -self.t_final * np.cos(self.swirl_freq * self.t_final)
        self.y_offset = -self.t_final * np.sin(self.swirl_freq * self.t_final)

        self.data = self._build()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"data": self.data[idx]}

    def scale_shift(self, data: np.ndarray):
        data[:, 0] -= data[:, 0].min()
        data[:, 1] -= data[:, 1].min()

        x_range = data[:, 0].ptp()
        y_range = data[:, 1].ptp()
        scale = x_range if x_range > y_range else y_range
        data /= scale

        return data

    def _build(self):
        # arm
        z = np.random.rand(self.size)
        arm = np.vstack((self.arm_lim * z, self.arm_lim * z)).T

        # swirl
        t = np.random.rand(self.size)
        t = np.sqrt(t) * self.t_final
        x = t * np.cos(self.swirl_freq * t) + self.x_offset
        y = t * np.sin(self.swirl_freq * t) + self.y_offset
        swirl = np.vstack((x, y)).T

        # combine
        final = np.where(np.random.rand(self.size, 1) > self.swirl_prop, arm, swirl)
        final = self.scale_shift(final)

        # add noise
        noise = np.random.normal(scale=self.std, size=(self.size, 2))
        final += noise

        # final score to ensure bounds
        final = self.scale_shift(final)

        return final


if __name__ == "__main__":
    ds = Lollipop(1000, std=0.02, swirl_prop=0.6)

    plt.scatter(x=ds[:, 0]["data"], y=ds[:, 1]["data"], s=1)
    plt.gca().set_aspect("equal")

    plt.show()
