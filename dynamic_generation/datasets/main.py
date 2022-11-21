from dynamic_generation.datasets.base import BaseDataModule
from dynamic_generation.datasets.lollipop import LollipopDataModule
from dynamic_generation.datasets.parity import ParityDataModule
from dynamic_generation.datasets.swiss_roll import SwissRollDataModule


def load_data_module(name: str, **dm_kwargs) -> BaseDataModule:
    match name:
        case "parity":
            dm = ParityDataModule(**dm_kwargs)
        case "swiss_roll":
            dm = SwissRollDataModule(**dm_kwargs)
        case "lollipop":
            dm = LollipopDataModule(**dm_kwargs)
        case other:
            raise ValueError(f"Not such dataset: {other}")

    return dm
