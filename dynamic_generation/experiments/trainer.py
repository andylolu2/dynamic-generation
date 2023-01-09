import tempfile
from logging import Formatter
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from absl import flags, logging
from ml_collections import FrozenConfigDict, config_flags
from torch import nn

from dynamic_generation.datasets.main import load_data_module
from dynamic_generation.types import Tensor, TrainState
from dynamic_generation.utils.actions import (
    Action,
    PeriodicEvalAction,
    PeriodicLogAction,
    PeriodicSaveAction,
    SaveConfigAction,
)
from dynamic_generation.utils.interrupt_handler import InterruptHandler
from dynamic_generation.utils.metrics import global_metrics
from dynamic_generation.utils.stats import confidence_interval
from dynamic_generation.utils.wandb import wandb_run

_CONFIG = config_flags.DEFINE_config_file("config")
flags.mark_flag_as_required("config")


class Trainer:
    def __init__(
        self,
        config,
        mode: Literal["train", "inference"] = "train",
        exp_dir: Path | None = None,
    ):
        self.config = config
        self.exp_dir = exp_dir

        # setup data type
        match config.precision:
            case "full":
                self.dtype = torch.float32
            case "mixed":
                self.dtype = torch.float16
            case other:
                raise ValueError(f"Unrecognised precision: {other}")

        logging.info(f"Running with precision: {config.precision}")
        logging.info(f"Running with dtype: {self.dtype}")

        # setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logging.info(f"Running with device: {self.device}")

        dm = load_data_module(**self.config.dataset.dm_kwargs)
        self.data_module = dm

        if mode == "train":
            # configure libraries
            formatter = Formatter(self.config.log.format, self.config.log.time_format)
            logging.get_absl_handler().setFormatter(formatter)  # type: ignore
            logging.set_verbosity(self.config.log.level)
            np.set_printoptions(precision=self.config.log.float_precision)
            plt.style.use("seaborn")

            # setup internal objects
            self.train_loader = dm.train_loader(**self.config.dataset.train_kwargs)
            self.eval_loader = dm.eval_loader(**self.config.dataset.eval_kwargs)
            self.actions = self.initialize_actions()
            self.interrupt_handler = InterruptHandler(handler=self.handle_interrupt)

            logging.info("\n" + str(self.config))

        self.train_state = self.initialize_state()

    @classmethod
    def run(cls, argv):
        """To be called by absl.app.run"""
        config: Any = FrozenConfigDict(_CONFIG.value)

        if config.benchmark.run:
            with tempfile.TemporaryDirectory() as temp_dir:
                trainer = cls(config, exp_dir=Path(temp_dir))
                trainer.benchmark()
        else:
            run = wandb_run(
                dry_run=config.dry_run,  # type: ignore
                project=config.project_name,
                config=config.to_dict(),
                tags=config.tags,
                notes=config.notes,
            )
            with run:
                exp_dir = Path("runs") / config.project_name / wandb.run.name
                trainer = cls(config, exp_dir=exp_dir)
                trainer.train()

    @property
    def train_step(self) -> int:
        return self.train_state["step"]

    def initialize_state(self) -> TrainState:
        """This function is expected to be subclassed"""
        return {"step": 0}

    def initialize_actions(self) -> list[Action]:
        assert self.exp_dir is not None

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
        save_config_action = SaveConfigAction(
            config=self.config,
            save_path=self.exp_dir / "config.yaml",
            dry_run=self.config.dry_run,
        )

        return [log_action, save_action, eval_action, save_config_action]

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

    def benchmark(self):
        logging.info("Warming up...")
        for _ in range(self.config.benchmark.warmup_steps):
            self._step(next(self.train_loader))

        logging.info("Running benchmark...")
        times = np.zeros(self.config.benchmark.steps, np.float32)
        with self.interrupt_handler as check_interrupt:
            for i in range(len(times)):
                start = perf_counter()
                check_interrupt()
                self._step(next(self.train_loader))

                times[i] = perf_counter() - start

        steps_per_s = 1 / times
        lo, hi = confidence_interval(steps_per_s)
        logging.info(f"Steps/s 95%: {lo:.4f} - {hi:.4f}s")

    def step(self):
        with global_metrics.capture("train"):
            start = perf_counter()
            self._step(next(self.train_loader))
            global_metrics.log("time_per_step", perf_counter() - start, "mean")

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
            return tuple(t.to(dtype=self.dtype, device=self.device) for t in tensors)

    def clip_grad(self, parameters, grad_norm_clip: float | None):
        if grad_norm_clip is not None:
            grad_norm = nn.utils.clip_grad.clip_grad_norm_(parameters, grad_norm_clip)
            global_metrics.log("grad_norm", grad_norm.item(), "mean")

    def save(self, path: Path):
        state_to_save = {}
        for k, v in self.train_state.items():
            if hasattr(v, "state_dict") and callable(v.state_dict):
                state_to_save[k] = v.state_dict()
            else:
                state_to_save[k] = v
        torch.save(state_to_save, path)

        logging.info(f"Saved states {list(state_to_save.keys())} to {path}")

    def load(self, path: Path):
        saved_state = torch.load(path, map_location=self.device)
        assert set(saved_state.keys()) == set(self.train_state.keys())

        for k, v in self.train_state.items():
            if hasattr(v, "load_state_dict") and callable(v.load_state_dict):
                v.load_state_dict(saved_state[k])
            else:
                self.train_state[k] = saved_state[k]

        logging.info(f"Restored states {list(saved_state.keys())} from {path}")

    def handle_interrupt(self):
        logging.info("Using default empty interrupt handler...")
