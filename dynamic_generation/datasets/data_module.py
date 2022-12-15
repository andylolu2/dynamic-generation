import abc
from typing import Iterable, Iterator

from dynamic_generation.types import Shape


class DataModule(abc.ABC):
    @abc.abstractmethod
    def train_loader(self, *args, **kwargs) -> Iterator:
        ...

    @abc.abstractmethod
    def eval_loader(self, *args, **kwargs) -> Iterable:
        ...

    @abc.abstractproperty
    def shape(self) -> dict[str, Shape]:
        ...
