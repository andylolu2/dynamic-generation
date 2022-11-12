from functools import partial
from pathlib import Path
from typing import Callable, Protocol

import wandb

from dynamic_generation.experiments.utils.logging import print_metrics
from dynamic_generation.experiments.utils.metrics import MetricsLogger


class Action(Protocol):
    def __call__(self, step: int) -> None:
        ...


class PeriodicAction(Action):
    """Runs a function periodically.

    Helper class that calls `self.action` every `interval` steps of training.
    """

    def __init__(self, interval: int):
        self.interval = interval

    def run(self, step: int):
        pass

    def __call__(self, step: int):
        if self.interval < 0:
            return
        elif step % self.interval == 0:
            self.run(step=step)


class PeriodicLogAction(PeriodicAction):
    def __init__(
        self, interval: int, metrics: MetricsLogger, group: str, dry_run: bool
    ):
        super().__init__(interval)
        self.metrics = metrics
        self.group = group
        self.dry_run = dry_run

    def run(self, step: int):
        metrics = self.metrics.collect(self.group)
        print_metrics(metrics, step)

        if not self.dry_run:
            wandb.log(metrics, step=step)


class PeriodicEvalAction(PeriodicAction):
    def __init__(
        self,
        interval: int,
        metrics: MetricsLogger,
        eval_fn: Callable[[], None],
        dry_run: bool,
    ):
        super().__init__(interval)
        self.metrics = metrics
        self.eval_fn = eval_fn
        self.dry_run = dry_run

    def run(self, step: int):
        with self.metrics.capture("eval"):
            self.eval_fn()
        metrics = self.metrics.collect(group="eval")
        print_metrics(metrics, step)

        if not self.dry_run:
            wandb.log(metrics, step=step)


class PeriodicSaveAction(PeriodicAction):
    def __init__(
        self,
        interval: int,
        save_dir: Path,
        save_ext: str,
        save_fn: Callable[[Path], None],
        dry_run: bool,
    ):
        super().__init__(interval)
        self.save_dir = save_dir
        self.save_ext = save_ext
        self.save_fn = save_fn
        self.dry_run = dry_run

    def run(self, step: int):
        if not self.dry_run:
            self.save_dir.mkdir(parents=True, exist_ok=True)

            # unlink old checkpoints
            for file in self.save_dir.glob(f"*{self.save_ext}"):
                file.unlink()

            # save new checkpoint
            file_name = str(step) + self.save_ext
            self.save_fn(self.save_dir / file_name)
