from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from absl import app, logging
from torch import nn
from torch.optim import Adam
from torchmetrics.functional.classification import binary_accuracy

from dynamic_generation.experiments.train_base import BaseTrainer, run_exp
from dynamic_generation.experiments.utils.actions import (
    Action,
    PeriodicEvalAction,
    PeriodicLogAction,
    PeriodicSaveAction,
)
from dynamic_generation.experiments.utils.metrics import weighted_binary_accuracy
from dynamic_generation.models.ponder_net import PonderNet, RnnPonderModule
from dynamic_generation.types import TrainState


class Trainer(BaseTrainer):
    @property
    def model(self) -> PonderNet:
        return self.train_state["model"]

    @property
    def optimizer(self) -> Adam:
        return self.train_state["optimizer"]

    def initialize_state(self) -> TrainState:
        state = super().initialize_state()

        def loss_fn(pred, target):
            pred, target = torch.broadcast_tensors(pred, target)
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            return loss.squeeze(-1)

        ponder_module = RnnPonderModule(**self.config.model.ponder_module_kwargs)
        model = PonderNet(
            ponder_module=ponder_module,
            loss_fn=loss_fn,
            trainer=self,
            **self.config.model.ponder_net_kwargs,
        )
        optimizer = Adam(model.parameters(), **self.config.optimizer_kwargs)

        model.to(device=self.device, dtype=self.dtype)

        state["model"] = model
        state["optimizer"] = optimizer
        return state

    def initialize_actions(self) -> list[Action]:
        log_action = PeriodicLogAction(
            interval=self.config.log_every,
            metrics=self.metrics,
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
            metrics=self.metrics,
            eval_fn=self.evaluate,
            dry_run=self.config.dry_run,
        )

        return [log_action, save_action, eval_action]

    def _step(self, item):
        with self.metrics.capture("train"):
            xs, ys_target = item["data"]
            xs, ys_target = self.cast(xs, ys_target)

            ys, ps = self.model(xs)
            loss = self.model.loss(ys_target=ys_target, ys=ys, ps=ps)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            # calculate weighted accuracy
            pred = ys.squeeze(-1)
            target = ys_target.broadcast_to(pred.shape)
            accuracy = weighted_binary_accuracy(pred, target, ps)

            self.log("accuracy", accuracy.item(), "mean")
            self.log("grad_norm", grad_norm.item(), "mean")
            if "epoch" in item:
                self.log("epoch", item["epoch"])

    @torch.inference_mode()
    def evaluate(self):
        logging.info("Begin evaluation...")

        self.model.eval()

        for item in self.eval_loader:
            xs, ys_target = item["data"]
            xs, ys_target = self.cast(xs, ys_target)
            ys, ps = self.model(xs)

            ys_idx = torch.multinomial(ps, num_samples=3, replacement=True)
            batch_idx = torch.arange(ys_idx.shape[0]).unsqueeze(-1)
            ys_sampled = ys[batch_idx, ys_idx]
            _ = self.model.loss(ys_target=ys_target, ys=ys, ps=ps)

            # calculate weighted accuracy
            pred = ys_sampled.squeeze(-1)
            target = ys_target.broadcast_to(pred.shape)
            accuracy = binary_accuracy(pred, target)
            self.log("accuracy", accuracy.item(), "mean")

        self.model.train()


def main(config):
    exp_dir = Path("runs") / config.project_name / wandb.run.name
    trainer = Trainer(config.trainer_config, exp_dir)

    if config.restore is not None:
        trainer.load(Path(config.restore))

    while config.steps < 0 or trainer.train_step < config.steps:
        trainer.step()


if __name__ == "__main__":
    app.run(lambda _: run_exp(main))
