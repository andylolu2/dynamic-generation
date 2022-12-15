from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence


class Schedule:
    @abstractmethod
    def __call__(self, step: int) -> float:
        ...


def linear_interpolate(x1, y1, x2, y2, t):
    assert 0 <= t <= 1
    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2
    return x, y


@dataclass
class StepSchedule(Schedule):
    milestones: Sequence[int]
    values: Sequence[float]

    def __post_init__(self):
        assert len(self.milestones) == len(self.values)
        assert sorted(self.milestones) == list(self.milestones)

    def __call__(self, step: int) -> float:
        ms = self.milestones
        vs = self.values

        i = 0
        while i < len(ms) and ms[i] < step:
            i += 1

        start_idx = max(i - 1, 0)
        end_idx = min(i, len(ms) - 1)

        x1, y1 = ms[start_idx], vs[start_idx]
        x2, y2 = ms[end_idx], vs[end_idx]

        if x1 == x2:
            return y1
        else:
            t = (step - x1) / (x2 - x1)
            _, y = linear_interpolate(x1, y1, x2, y2, t)
            return y


class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, step: int) -> float:
        return self.value


def load_schedule(name: str, *args, **kwargs):
    match name:
        case "step":
            return StepSchedule(*args, **kwargs)
        case "constant":
            return ConstantSchedule(*args, **kwargs)
        case _:
            raise ValueError(f"No such schedule: {name}")
