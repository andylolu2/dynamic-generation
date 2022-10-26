from typing import Iterator

from torch.utils.data import DataLoader


def infinite_loader(ds_loader: DataLoader) -> Iterator:
    epoch = 0
    while True:
        # automatically reshuffles on every iteration
        for item in ds_loader:
            yield epoch, item
        epoch += 1
