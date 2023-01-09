from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from absl import logging
from torch.utils.data import DataLoader, Subset

from dynamic_generation.datasets.data_module import DataModule
from dynamic_generation.datasets.utils import infinite_loader
from dynamic_generation.types import Shape

NUM_CLASSES = 10


@dataclass
class MNISTDataModule(DataModule):
    data_path: Path
    size: int
    range_: tuple[float, float]

    def __post_init__(self):
        lo, hi = self.range_
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),  # range: [0,1]
                torchvision.transforms.Normalize(
                    mean=[-lo / (hi - lo)],
                    std=[1 / (hi - lo)],
                ),  # range: self.range_
                torchvision.transforms.Resize((self.size, self.size)),
            ]
        )

    @property
    def shape(self) -> dict[str, Shape]:
        return {"x": (1, self.size, self.size), "y": (NUM_CLASSES,)}

    def train_loader(self, batch_size: int, size: int) -> Iterator:
        ds = MNISTDataset(
            str(self.data_path), transform=self.transform, train=False, download=True
        )
        if size > 0:
            indices = np.random.choice(len(ds), size, replace=False).tolist()
            ds = Subset(ds, indices)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        loader = infinite_loader(loader)
        return loader

    def eval_loader(self, batch_size: int, size: int) -> Iterable:
        ds = MNISTDataset(
            str(self.data_path), transform=self.transform, train=False, download=True
        )
        if 0 < size < len(ds):
            indices = np.random.choice(len(ds), size, replace=False).tolist()
            ds = Subset(ds, indices)
        if size > len(ds):
            logging.warning(
                f"Asking for {size} items in eval set but there is only {len(ds)} items."
            )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader


class MNISTDataset(torchvision.datasets.MNIST):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return {
            "x": x,
            "y": F.one_hot(torch.tensor(y), num_classes=NUM_CLASSES),
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from dynamic_generation.utils.image import tile_image

    dm = MNISTDataModule(Path("data"), 32)
    loader = dm.train_loader(9, -1)

    x = next(loader)["x"]
    img = tile_image(x)
    plt.imshow(img)
    plt.show()
