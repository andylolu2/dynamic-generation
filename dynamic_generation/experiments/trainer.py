from logging import Formatter
from pathlib import Path
from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from absl import flags, logging
from ml_collections import FrozenConfigDict, config_flags

from dynamic_generation.datasets.base import BaseDataModule
from dynamic_generation.datasets.main import load_data_module
from dynamic_generation.experiments.utils.actions import (
    Action,
    PeriodicEvalAction,
    PeriodicLogAction,
    PeriodicSaveAction,
)
from dynamic_generation.experiments.utils.wandb import wandb_run
from dynamic_generation.types import Tensor, TrainState
from dynamic_generation.utils.interrupt_handler import InterruptHandler

_CONFIG = config_flags.DEFINE_config_file("config")
flags.mark_flag_as_required("config")


class Trainer:
    def __init__(self, config, exp_dir: Path):
        self.config = config
        self.exp_dir = exp_dir
        self.dtype = torch.float32

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_state = self.initialize_state()
        self.actions = self.initialize_actions()

        # setup data loaders
        dm = load_data_module(**self.config.dataset.dm_kwargs)
        self.data_module = dm
        self.train_loader = dm.train_loader(**self.config.dataset.train_kwargs)
        self.eval_loader = dm.eval_loader(**self.config.dataset.eval_kwargs)

        # handle keyboard interrupts
        self.interrupt_handler = InterruptHandler(handler=self.handle_interrupt)

        # configure logging
        formatter = Formatter(self.config.log.format, self.config.log.time_format)
        logging.get_absl_handler().setFormatter(formatter)  # type: ignore
        logging.set_verbosity(self.config.log.level)
        np.set_printoptions(precision=self.config.log.float_precision)

        # configure plotting
        plt.style.use("seaborn")

        # logs
        logging.info(f"Running with device: {self.device}")
        logging.info(self.config)

    @classmethod
    def run(cls, argv):
        """To be called by absl.app.run"""
        config: Any = FrozenConfigDict(_CONFIG.value)
        run = wandb_run(
            dry_run=config.dry_run,  # type: ignore
            project=config.project_name,
            config=config.to_dict(),
            tags=config.tags,
            notes=config.notes,
        )
        with run:
            exp_dir = Path("runs") / config.project_name / wandb.run.name
            trainer = cls(config, exp_dir)
            trainer.train()

    @property
    def train_step(self) -> int:
        return self.train_state["step"]

    def initialize_state(self) -> TrainState:
        """This function is expected to be subclassed"""
        return {"step": 0}

    def initialize_data_module(self) -> BaseDataModule:
        """This function is expected to be subclassed"""
        raise NotImplementedError()

    def initialize_actions(self) -> list[Action]:
        log_action = PeriodicLogAction(
            interval=self.config.log.every,
            group="train",
            dry_run=self.config.dry_run,
        )
        save_action = PeriodicSaveAction(
            interval=self.config.save.every,
            save_dir=self.exp_dir / self.config.save.dir,
            save_ext=self.config.save.ext,
            save_fn=self.save,
            dry_run=self.config.dry_run,
        )
        eval_action = PeriodicEvalAction(
            interval=self.config.eval.every,
            eval_fn=self.evaluate,
            dry_run=self.config.dry_run,
        )

        return [log_action, save_action, eval_action]

    def _step(self, item):
        """This function is expected to be subclassed"""
        raise NotImplementedError()

    def evaluate(self):
        """This function is expected to be subclassed"""
        raise NotImplementedError()

    def train(self):
        if self.config.restore is not None:
            self.load(Path(self.config.restore))

        with self.interrupt_handler as check_interrupt:
            while self.config.steps < 0 or self.train_step <= self.config.steps:
                check_interrupt()
                self.step()

    def step(self):
        self._step(next(self.train_loader))
        for action in self.actions:
            action(self.train_step)
        self.train_state["step"] += 1

    @overload
    def cast(self, __tensor: Tensor) -> Tensor:
        ...

    @overload
    def cast(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        ...

    def cast(self, *tensors: Tensor):
        if len(tensors) == 1:
            return tensors[0].to(dtype=self.dtype, device=self.device)
        else:
            return tuple(self.cast(t) for t in tensors)

    def save(self, path: Path):
        state_to_save = {}
        for k, v in self.train_state.items():
            if hasattr(v, "state_dict") and callable(v.state_dict):
                state_to_save[k] = v.state_dict()
            else:
                state_to_save[k] = v
        torch.save(state_to_save, path)

        logging.info(f"Saved states {list(state_to_save.keys())} to {path}")

    def load(self, path: Path, map_location: str | None = None):
        saved_state = torch.load(path, map_location=map_location)
        assert set(saved_state.keys()) == set(self.train_state.keys())

        for k, v in self.train_state.items():
            if hasattr(v, "load_state_dict") and callable(v.load_state_dict):
                v.load_state_dict(saved_state[k])
            else:
                self.train_state[k] = saved_state[k]

        logging.info(f"Restored states {list(saved_state.keys())} from {path}")

    def handle_interrupt(self):
        logging.info("Using default empty interrupt handler...")