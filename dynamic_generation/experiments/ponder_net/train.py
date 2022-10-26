from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F
from absl import app, logging
from torch import nn
from torch.optim import Adam
from torchmetrics.functional.classification import binary_accuracy

import wandb
from dynamic_generation.datasets.main import load_dataset
from dynamic_generation.experiments.train_base import BaseTrainer, run_exp
from dynamic_generation.experiments.utils.actions import Action, periodic
from dynamic_generation.experiments.utils.logging import print_metrics
from dynamic_generation.experiments.utils.metrics import weighted_binary_accuracy
from dynamic_generation.models.ponder_net import PonderNet, RnnPonderModule
from dynamic_generation.types import TrainState


class Trainer(BaseTrainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

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

    def initialize_dataloader(self) -> tuple[Iterator, Iterable]:
        return load_dataset("parity", **self.config.dataset_kwargs)

    def initialize_actions(self) -> list[Action]:
        @periodic(self.config.log_every)
        def log_action(step: int):
            metrics = self.metrics.collect(group="train")
            print_metrics(metrics, step)
            if not self.config.dry_run:
                wandb.log(metrics, step=step)

        @periodic(self.config.save_every)
        def save_action(step: int):
            if not self.config.dry_run:
                save_dir = self.exp_dir / self.config.save.dir
                save_dir.mkdir(parents=True, exist_ok=True)

                # unlink old checkpoints
                for file in save_dir.glob(f"*{self.config.save.ext}"):
                    file.unlink()

                # save new checkpoint
                file_name = str(step) + self.config.save.ext
                self.save(save_dir / file_name)

        @periodic(self.config.eval_every)
        def eval_action(step: int):
            with self.metrics.capture("eval"):
                self.evaluate()
            metrics = self.metrics.collect(group="eval")
            print_metrics(metrics, step)
            if not self.config.dry_run:
                wandb.log(metrics, step=step)

        return [log_action, save_action, eval_action]

    @property
    def model(self) -> PonderNet:
        return self.train_state["model"]

    @property
    def optimizer(self) -> Adam:
        return self.train_state["optimizer"]

    def _step(self, item):
        with self.metrics.capture("train"):
            xs, ys_target = item
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

    @torch.inference_mode()
    def evaluate(self):
        logging.info("Begin evaluation...")

        self.model.eval()

        for xs, ys_target in self.eval_loader:
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
