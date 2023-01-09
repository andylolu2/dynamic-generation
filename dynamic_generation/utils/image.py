import math

import torchvision

from dynamic_generation.types import Tensor


def tile_image(
    x: Tensor,
    nrow: int | None = None,
    input_range: tuple[float, float] = (0.0, 1.0),
):
    # x: N,C,H,W
    if nrow is None:
        nrow = math.floor(math.sqrt(x.shape[0]))
    else:
        nrow = x.shape[0] // nrow

    # img_grid: C,H',W'
    img_grid = torchvision.utils.make_grid(x, nrow=nrow, pad_value=1.0)

    lo, hi = input_range
    img_grid = (img_grid - lo) / (hi - lo)  # maps to [0, 1]
    return torchvision.transforms.ToPILImage()(img_grid)
