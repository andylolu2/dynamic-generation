from typing import Any, ParamSpec, TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor
TrainState: TypeAlias = dict[str, Any]
Metrics: TypeAlias = dict[str, Any]
P = ParamSpec("P")
