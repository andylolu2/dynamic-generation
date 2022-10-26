import abc
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch
from torchtyping import TensorType


def weighted_binary_accuracy(
    pred: TensorType["batch", "samples"],
    target: TensorType["batch", "samples"],
    weight: TensorType["batch", "samples"],
    logits: bool = True,
    threshold: float = 0.5,
):
    assert pred.shape == target.shape == weight.shape

    # check if `weight` is a valid distribution for each batch
    weight_sum = weight.sum(-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum))

    if logits:
        pred = pred.sigmoid()

    pred_labels = pred > threshold
    target_labels = target.bool()
    correct = pred_labels == target_labels

    weighted_correct = weight * correct
    weighted_acc = weighted_correct.sum(-1).mean()

    return weighted_acc


class BaseAccumulator(abc.ABC):
    @abc.abstractmethod
    def __init__(self, init_value):
        ...

    @abc.abstractmethod
    def update(self, value):
        ...

    @abc.abstractmethod
    def compute(self):
        ...


class ReplaceAccumulator(BaseAccumulator):
    def __init__(self, init_value):
        self.value = init_value

    def update(self, value):
        self.value = value

    def compute(self):
        return self.value


class MeanAccumulator(BaseAccumulator):
    def __init__(self, init_value):
        self.value = torch.as_tensor(init_value)
        self.count = torch.tensor(1.0)

    def update(self, value):
        self.value = self.value + value
        self.count = self.count + 1

    def compute(self):
        res = self.value / self.count
        if res.numel() == 1:
            res = res.squeeze()
        return res.numpy()


class SumAccumulator(BaseAccumulator):
    def __init__(self, init_value):
        self.value = torch.as_tensor(init_value)

    def update(self, value):
        self.value = self.value + value

    def compute(self):
        return self.value.numpy()


class MetricsLogger:
    def __init__(self):
        self.logs: dict[str, dict[str, BaseAccumulator]] = defaultdict(dict)
        self.current_group: str | None = None

    @contextmanager
    def capture(self, group_name: str):
        tmp = self.current_group
        self.current_group = group_name
        yield
        self.current_group = tmp

    def log(self, k: str, v: Any, acc="replace", group: str | None = None):
        if group is None:
            group = self.current_group
        assert group is not None
        if k not in self.logs:
            match acc:
                case "replace":
                    acc_cls = ReplaceAccumulator
                case "mean":
                    acc_cls = MeanAccumulator
                case "sum":
                    acc_cls = SumAccumulator
                case _:
                    raise ValueError()
            self.logs[group][k] = acc_cls(v)
        else:
            self.logs[group][k].update(v)

    def collect(self, group: str):
        res = {}
        for k, v in self.logs[group].items():
            res[f"{group}/{k}"] = v.compute()
        return res

    def clear(self, group: str):
        self.logs.pop(group, None)
