import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

from dynamic_generation.utils.accumulators import (
    Accumulator,
    MaxAccumulator,
    MeanAccumulator,
    MinAccumulator,
    ReplaceAccumulator,
    StackAccumulator,
    SumAccumulator,
)


class MetricsLogger:
    def __init__(self):
        self.logs: dict[str, dict[str, Accumulator]] = defaultdict(dict)
        self.current_group: str | None = None

    @contextmanager
    def capture(self, group_name: str | None):
        tmp = self.current_group
        self.current_group = group_name
        yield
        self.current_group = tmp

    def log(self, k: str, v: Any, acc="replace", group: str | None = None, **kwargs):
        if group is None:
            group = self.current_group

        if group is None:
            warnings.warn(f"{k} is not being captured", UserWarning)
            return

        if k not in self.logs[group]:
            match acc:
                case "replace":
                    acc_cls = ReplaceAccumulator
                case "mean":
                    acc_cls = MeanAccumulator
                case "sum":
                    acc_cls = SumAccumulator
                case "max":
                    acc_cls = MaxAccumulator
                case "min":
                    acc_cls = MinAccumulator
                case "stack":
                    acc_cls = StackAccumulator
                case _:
                    raise ValueError()
            self.logs[group][k] = acc_cls(v, **kwargs)
        else:
            self.logs[group][k].update(v)

    def collect(self, group: str, clear=True):
        res = {}
        for k, v in self.logs[group].items():
            res[f"{group}/{k}"] = v.compute()

        if clear:
            self.clear(group)
        return res

    def clear(self, group: str):
        self.logs.pop(group, None)


global_metrics = MetricsLogger()
