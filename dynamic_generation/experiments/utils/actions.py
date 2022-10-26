from functools import partial
from typing import Callable, Protocol


class Action(Protocol):
    def __call__(self, step: int) -> None:
        ...


class PeriodicAction(Action):
    """Runs a function periodically.

    Helper class that calls `self.run` every `interval` steps of training.
    """

    def __init__(self, interval: int, action: Action):
        self.interval = interval
        self.action = action

    def __call__(self, step: int):
        if self.interval < 0:
            return
        elif step % self.interval == 0:
            self.action(step=step)


def periodic(interval: int) -> Callable[[Action], PeriodicAction]:
    return partial(PeriodicAction, interval)
