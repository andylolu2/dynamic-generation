import torch

from dynamic_generation.types import Tensor

EPSILON = 1e-7


def safe_log(x: Tensor):
    return (x + EPSILON).log()


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
