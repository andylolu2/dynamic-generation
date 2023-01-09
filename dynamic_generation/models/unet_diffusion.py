from dynamic_generation.types import Tensor

from .diffusion import Diffuser
from .unet import UNet


class UNetDiffuser(Diffuser):
    def __init__(self, unet_kwargs: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unet = UNet(self.input_dims, **unet_kwargs)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.unet.forward(x_t, t)
