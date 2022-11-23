import abc
from typing import Iterable, Iterator


class BaseDataModule(abc.ABC):
    @abc.abstractmethod
    def train_loader(self, *args, **kwargs) -> Iterator:
        ...

    @abc.abstractmethod
    def eval_loader(self, *args, **kwargs) -> Iterable:
        ...
