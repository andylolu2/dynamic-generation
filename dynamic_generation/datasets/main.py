from typing import Iterable, Iterator

from dynamic_generation.datasets.base import BaseDataModule
from dynamic_generation.datasets.parity import ParityDataModule
from dynamic_generation.datasets.swiss_roll import SwissRollDataModule


def load_dataset(
    name: str,
    ds_kwargs,
    train_kwargs,
    eval_kwargs,
) -> tuple[Iterator, Iterable]:

    dm: BaseDataModule
    match name:
        case "parity":
            dm = ParityDataModule(**ds_kwargs)
        case "swiss_roll":
            dm = SwissRollDataModule(**ds_kwargs)
        case other:
            raise ValueError(f"Not such dataset: {other}")

    train_loader = dm.train_loader(**train_kwargs)
    eval_loader = dm.eval_loader(**eval_kwargs)

    return train_loader, eval_loader
