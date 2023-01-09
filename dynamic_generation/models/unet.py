from abc import abstractmethod
from functools import partial
from typing import Collection, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn

from dynamic_generation.types import Shape, Tensor

from .diffusion import TimeEmbedding

KERNEL_SIZE = 3
EPS_FP32 = 1e-5
EPS_FP16 = 1e-3


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x: Tensor, time_emb: Tensor):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Utility class to inject the time embeddings."""

    def forward(self, x: Tensor, time_emb: Tensor):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Sequential):
    def __init__(self, dim: int, dim_out: int | None = None):
        if dim_out is None:
            dim_out = dim
        super().__init__(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, dim_out, KERNEL_SIZE, padding="same"),
        )


class Downsample(nn.Sequential):
    def __init__(self, dim: int, dim_out: int | None = None):
        if dim_out is None:
            dim_out = dim
        super().__init__(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim * 4, dim_out, 1),
        )


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x: Tensor):
        eps = EPS_FP32 if x.dtype == torch.float32 else EPS_FP16

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _Block(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, groups: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.groups = groups

        self.projection = WeightStandardizedConv2d(
            dim_in, dim_out, KERNEL_SIZE, padding="same"
        )
        self.group_norm = nn.GroupNorm(groups, dim_out)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor, scale_shift: tuple[Tensor, Tensor] | None = None):
        x = self.projection(x)
        x = self.group_norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x


class ResNetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim_in: int, dim_out: int, groups: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.groups = groups

        self.block_1 = _Block(dim_in, dim_out, groups=groups)
        self.block_2 = _Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.block_1(x)
        h = self.block_2(h)
        return h + self.res_conv(x)


class TimeResNetBlock(ResNetBlock, TimestepBlock):
    def __init__(self, dim_time: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_time = dim_time
        self.ff = nn.Linear(dim_time, self.dim_out * 2)

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        time_emb = self.ff(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block_1(x, scale_shift=scale_shift)
        h = self.block_2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    """https://arxiv.org/abs/1706.03762"""

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h d (x y)", h=self.heads), qkv
        )
        q = q / self.dim_head**0.5

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearSelfAttention(nn.Module):
    """https://arxiv.org/abs/1812.01243"""

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head

        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q / self.dim_head**0.5
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class UNet(nn.Module):
    def __init__(
        self,
        input_dims: Shape,
        dim_base: int,
        dim_time: int,
        blocks_per_level: int,
        dim_mults: Sequence[int],
        attention_resolutions: Collection[int],
        time_emb_kwargs: dict,
        resnet_block_kwargs: dict,
        self_attention_kwargs: dict,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.dim_base = dim_base
        self.dim_mults = dim_mults
        self.blocks_per_level = blocks_per_level
        self.attention_resolutions = attention_resolutions

        # determine dimensions
        c, h, w = input_dims

        time_emb = TimeEmbedding(**time_emb_kwargs)
        self.time_embed = nn.Sequential(
            time_emb,
            nn.Linear(time_emb.dim, dim_time),
            nn.SiLU(),
            nn.Linear(dim_time, dim_time),
            nn.SiLU(),
        )

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(
            TimestepEmbedSequential(nn.Conv2d(c, dim_base, 1, padding="same"))
        )

        input_block_dims = [dim_base]
        dim_prev = dim_base  # used to track the previous layer's output dim
        ds = 1

        for level, mult in enumerate(dim_mults):
            dim = mult * dim_base
            for _ in range(blocks_per_level):
                block = TimestepEmbedSequential()
                block.append(
                    TimeResNetBlock(
                        dim_time=dim_time,
                        dim_in=dim_prev,
                        dim_out=dim,
                        **resnet_block_kwargs
                    )
                )
                dim_prev = dim
                if ds in attention_resolutions:
                    block.append(SelfAttention(dim, **self_attention_kwargs))
                self.input_blocks.append(block)
                input_block_dims.append(dim)

            if level != len(dim_mults) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(dim)))
                input_block_dims.append(dim)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            TimeResNetBlock(
                dim_time=dim_time,
                dim_in=dim_prev,
                dim_out=dim_prev,
                **resnet_block_kwargs
            ),
            SelfAttention(dim_prev, **self_attention_kwargs),
            TimeResNetBlock(
                dim_time=dim_time,
                dim_in=dim_prev,
                dim_out=dim_prev,
                **resnet_block_kwargs
            ),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(dim_mults))):
            dim = mult * dim_base
            for i in range(blocks_per_level + 1):
                block = TimestepEmbedSequential()
                block.append(
                    TimeResNetBlock(
                        dim_time=dim_time,
                        dim_in=dim_prev + input_block_dims.pop(),
                        dim_out=dim,
                        **resnet_block_kwargs
                    )
                )
                dim_prev = dim
                if ds in attention_resolutions:
                    block.append(SelfAttention(dim, **self_attention_kwargs))

                if level != 0 and i == blocks_per_level:
                    block.append(Upsample(dim_prev))
                    ds //= 2
                self.output_blocks.append(block)

        self.out = nn.Sequential(
            ResNetBlock(dim_in=dim_prev, dim_out=dim_prev, **resnet_block_kwargs),
            nn.Conv2d(dim_prev, c, 1, padding="same"),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        time_emb = self.time_embed(t)

        h = x
        hs = []
        for module in self.input_blocks:
            h = module(h, time_emb)
            hs.append(h)
        h = self.middle_block(h, time_emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)  # cat on the channel dimension
            h = module(cat_in, time_emb)

        return self.out(h)
