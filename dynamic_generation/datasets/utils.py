from typing import Iterator

from torch.utils.data import DataLoader

from dynamic_generation.utils.metrics import global_metrics


def infinite_loader(ds_loader: DataLoader) -> Iterator:
    epoch = 0
    while True:
        # automatically reshuffles on every iteration
        for item in ds_loader:
            global_metrics.log("epoch", epoch)
            item["epoch"] = epoch
            yield item
        epoch += 1
