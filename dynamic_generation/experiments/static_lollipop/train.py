from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from absl import app
from torch.optim import Adam

from dynamic_generation.experiments.train_base import BaseTrainer, run_exp
from dynamic_generation.experiments.utils.actions import (
    Action,
    PeriodicEvalAction,
    PeriodicLogAction,
    PeriodicSaveAction,
)
from dynamic_generation.experiments.utils.metrics import global_metrics
from dynamic_generation.models.toy_generator import ToyGenerator
from dynamic_generation.types import TrainState


class Trainer(BaseTrainer):
    @property
    def model(self) -> ToyGenerator:
        return self.train_state["model"]

    @property
    def optimizer(self) -> Adam:
        return self.train_state["optimizer"]

    def initialize_state(self) -> TrainState:
        train_state = super().initialize_state()
        model = ToyGenerator(trainer=self, **self.config.model_kwargs)
        optimizer = Adam(model.parameters(), **self.config.optimizer_kwargs)

        model.to(device=self.device, dtype=self.dtype)

        train_state["model"] = model
        train_state["optimizer"] = optimizer
        return train_state

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
            x = item["data"]
            x = self.cast(x)
            loss = self.model.loss(x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if "epoch" in item:
                global_metrics.log("epoch", item["epoch"], "replace")

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()

        xs = []
        for item in self.eval_loader:
            x = item["data"]
            x = self.cast(x)
            xs.append(x)
            _ = self.model.loss(x)

        real = torch.stack(xs).cpu().numpy()
        real = real.reshape(-1, real.shape[-1])

        # generate samples
        x_hat = self.model.generate(self.config.eval.samples, self.device)
        data = x_hat.cpu().numpy()

        # plot results
        fig, ax = plt.subplots()
        ax.scatter(x=real[:, 0], y=real[:, 1], s=1, c="r", zorder=1, label="Real")
        ax.scatter(x=data[:, 0], y=data[:, 1], s=1, c="b", zorder=2, label="Generated")
        ax.set(box_aspect=1)
        fig.tight_layout()

        global_metrics.log("samples", wandb.Image(fig), "replace")

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
