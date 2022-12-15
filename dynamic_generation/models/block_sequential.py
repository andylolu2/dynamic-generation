from torch import nn

from .blocks import Block


class BlockSequential(nn.Sequential):
    def __init__(
        self,
        dims: list[int],
        block: Block,
        first_block: Block | None = None,
        last_block: Block | None = None,
    ):
        super().__init__()

        n_layers = len(dims) - 1
        assert n_layers >= 1

        if first_block is None:
            first_block = block

        if last_block is None:
            last_block = block

        for i in range(n_layers):
            if i == 0:
                current_block = first_block
            elif i == n_layers - 1:
                current_block = last_block
            else:
                current_block = block

            self.append(current_block.new(dims[i], dims[i + 1]))
