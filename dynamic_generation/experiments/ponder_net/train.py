from pathlib import Path

import torch
import torch.distributions as D
import wandb
from absl import app, logging
from torch import nn
from torch.optim import Optimizer
from torchmetrics.functional.classification import binary_accuracy

from dynamic_generation.experiments.train_base import BaseTrainer, run_exp
from dynamic_generation.experiments.utils.actions import (
    Action,
    PeriodicEvalAction,
    PeriodicLogAction,
    PeriodicSaveAction,
)
from dynamic_generation.experiments.utils.metrics import global_metrics
from dynamic_generation.experiments.utils.optimizers import load_optimizer
from dynamic_generation.models.ponder_module import RNNPonderModule
from dynamic_generation.models.ponder_net import PonderNet
from dynamic_generation.types import TrainState


class Trainer(BaseTrainer):
    @property
    def model(self) -> PonderNet:
        return self.train_state["model"]

    @property
    def optimizer(self) -> Optimizer:
        return self.train_state["optimizer"]

    def initialize_state(self) -> TrainState:
        state = super().initialize_state()

        ponder_module = RNNPonderModule(**self.config.model.ponder_module_kwargs)
        model = PonderNet(
            ponder_module=ponder_module, **self.config.model.ponder_net_kwargs
        )
        optimizer = load_optimizer(
            params=model.parameters(), **self.config.optimizer_kwargs
        )

        model.to(device=self.device, dtype=self.dtype)

        state["model"] = model
        state["optimizer"] = optimizer
        return state

    def initialize_actions(self) -> list[Action]:
        log_action = PeriodicLogAction(
            interval=self.config.log_every,
            group="train",
            dry_run=self.config.dry_run,
        )
        save_action = PeriodicSaveAction(
            interval=self.config.save_every,
            save_dir=self.exp_dir / self.config.save.dir,
            save_ext=self.config.save.ext,
            save_fn=self.save,
            dry_run=self.config.dry_run,
        )
        eval_action = PeriodicEvalAction(
            interval=self.config.eval_every,
            eval_fn=self.evaluate,
            dry_run=self.config.dry_run,
        )

        return [log_action, save_action, eval_action]

    def _step(self, item):
        with global_metrics.capture("train"):
            x, y_true = item["data"]
            x, y_true = self.cast(x, y_true)

            y, halt_dist = self.model(x)
            y_pred = D.MixtureSameFamily(
                mixture_distribution=halt_dist,
                component_distribution=D.Bernoulli(logits=y.squeeze(-1)),
            )
            loss = self.model.loss(y_true=y_true, y_pred=y_pred, halt_dist=halt_dist)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            pred = y_pred.sample((self.config.train.ponder_samples,))  # type: ignore
            target, pred = torch.broadcast_tensors(y_true, pred)
            accuracy = binary_accuracy(pred, target)

            global_metrics.log("accuracy", accuracy.item(), "mean")
            global_metrics.log("grad_norm", grad_norm.item(), "mean")
            if "epoch" in item:
                global_metrics.log("epoch", item["epoch"])

    @torch.inference_mode()
    def evaluate(self):
        logging.info("Begin evaluation...")

        self.model.eval()

        for item in self.eval_loader:
            x, y_true = item["data"]
            x, y_true = self.cast(x, y_true)

            y, halt_dist = self.model(x)
            y_pred = D.MixtureSameFamily(
                mixture_distribution=halt_dist,
                component_distribution=D.Bernoulli(logits=y.squeeze(-1)),
            )
            _ = self.model.loss(y_true=y_true, y_pred=y_pred, halt_dist=halt_dist)

            # calculate weighted accuracy
            pred = y_pred.sample((self.config.eval.ponder_samples,))  # type: ignore
            target, pred = torch.broadcast_tensors(y_true, pred)
            accuracy = binary_accuracy(pred, target)
            global_metrics.log("accuracy", accuracy.item(), "mean")

        self.model.train()


def main(config):
    exp_dir = Path("runs") / config.project_name / wandb.run.name
    trainer = Trainer(config.trainer_config, exp_dir)

    if config.restore is not None:
        trainer.load(Path(config.restore))

    with trainer.interrupt_handler as check_interrupt:
        while config.steps < 0 or trainer.train_step < config.steps:
            check_interrupt()
            trainer.step()


if __name__ == "__main__":
    app.run(lambda _: run_exp(main))
