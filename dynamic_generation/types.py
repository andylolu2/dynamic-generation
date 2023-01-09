from typing import Any, ParamSpec, TypeAlias

import torch
from torch.types import _size

Tensor: TypeAlias = torch.Tensor
TrainState: TypeAlias = dict[str, Any]
Metrics: TypeAlias = dict[str, Any]
P = ParamSpec("P")
Shape: TypeAlias = _size
