from dataclasses import dataclass
from typing import Protocol

from torch import nn


class BlockMaker(Protocol):
    def __call__(self, in_dim: int, out_dim: int) -> nn.Module:
        ...


class Block:
    def new(self, in_dim: int, out_dim: int):
        return nn.Sequential()


@dataclass
class LinearBlock(Block):
    pre_block: nn.Module | None = None
    post_block: nn.Module | None = None

    def new(self, in_dim: int, out_dim: int):
        block = super().new(in_dim, out_dim)

        if self.pre_block is not None:
            block.append(self.pre_block)

        block.append(nn.Linear(in_dim, out_dim))

        if self.post_block is not None:
            block.append(self.post_block)

        return block
