from contextlib import contextmanager
from logging import Formatter
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
from absl import flags, logging
from ml_collections import FrozenConfigDict, config_flags

from dynamic_generation.experiments.utils.actions import Action
from dynamic_generation.experiments.utils.metrics import MetricsLogger
from dynamic_generation.experiments.utils.wandb import wandb_run
from dynamic_generation.types import Tensor, TrainState

_CONFIG = config_flags.DEFINE_config_file("config")
flags.mark_flag_as_required("config")


def run_exp(main_fn: Callable[[FrozenConfigDict], None]):
    config = FrozenConfigDict(_CONFIG.value)

    # configure logging
    formatter = Formatter(config.logging.format, config.logging.time_format)
    logging.get_absl_handler().setFormatter(formatter)  # type: ignore
    logging.set_verbosity(config.logging.level)
    np.set_printoptions(precision=config.logging.float_precision)

    logging.info(config)

    with wandb_run(config.project_name, config.dry_run, config):  # type: ignore
        main_fn(config)


class BaseTrainer:
    def __init__(self, config, exp_dir: Path):
        self.config = config
        self.exp_dir = exp_dir
        self.dtype = torch.float32

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_state = self.initialize_state()
        self.train_loader, self.eval_loader = self.initialize_dataloader()
        self.actions = self.initialize_actions()
        self.metrics = MetricsLogger()
        self.log = self.metrics.log

        # logs
        logging.info(f"Running with device: {self.device}")

    @property
    def train_step(self) -> int:
        return self.train_state["step"]

    def initialize_state(self) -> TrainState:
        """This function is expected to be subclassed"""
        return {"step": 0}

    def initialize_dataloader(self) -> tuple[Iterator, Iterable]:
        """This function is expected to be subclassed"""
        raise NotImplementedError()

    def initialize_actions(self) -> list[Action]:
        """This function is expected to be subclassed"""
        return []

    def _step(self, item):
        """This function is expected to be subclassed"""
        raise NotImplementedError()

    def evaluate(self):
        """This function is expected to be subclassed"""
        raise NotImplementedError()

    def step(self):
        self._step(next(self.train_loader))
        for action in self.actions:
            action(self.train_step)
        self.train_state["step"] += 1

    def cast(self, *tensors: Tensor):
        return tuple(t.to(dtype=self.dtype, device=self.device) for t in tensors)

    def save(self, path: Path):
        state_to_save = {}
        for k, v in self.train_state.items():
            if hasattr(v, "state_dict") and callable(v.state_dict):
                state_to_save[k] = v.state_dict()
            else:
                state_to_save[k] = v
        torch.save(state_to_save, path)

    def load(self, path: Path, map_location: str | None = None):
        saved_state = torch.load(path, map_location=map_location)
        assert set(saved_state.keys()) == set(self.train_state.keys())

        for k, v in self.train_state.items():
            if hasattr(v, "load_state_dict") and callable(v.load_state_dict):
                v.load_state_dict(saved_state[k])
            else:
                self.train_state[k] = saved_state[k]
