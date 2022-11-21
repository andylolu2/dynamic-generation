from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from absl import app
from torch import nn
from torch.optim import Adam

from dynamic_generation.experiments.train_base import BaseTrainer, run_exp
from dynamic_generation.experiments.utils.actions import (
    Action,
    PeriodicEvalAction,
    PeriodicLogAction,
    PeriodicSaveAction,
)
from dynamic_generation.experiments.utils.metrics import global_metrics
from dynamic_generation.experiments.utils.schedules import load_schedule
from dynamic_generation.models.vae import UniformBetaVAE
from dynamic_generation.types import TrainState


class Trainer(BaseTrainer):
    def __init__(self, config, exp_dir: Path):
        super().__init__(config, exp_dir)
        self.beta_schedule = load_schedule(**self.config.train.beta_schedule_kwargs)

    @property
    def model(self) -> UniformBetaVAE:
        return self.train_state["model"]

    @property
    def optimizer(self) -> Adam:
        return self.train_state["optimizer"]

    def initialize_state(self) -> TrainState:
        train_state = super().initialize_state()
        model = UniformBetaVAE(**self.config.model_kwargs)
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
            out = self.model(x)

            beta = self.beta_schedule(self.train_step)
            loss = self.model.loss(x, out, beta)

            grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            global_metrics.log("grad_norm", grad_norm.item(), "mean")
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
            out = self.model(x)

            beta = self.beta_schedule(self.train_step)
            _ = self.model.loss(x, out, beta)

        real = torch.stack(xs).cpu().numpy()
        real = real.reshape(-1, real.shape[-1])

        # generate samples
        x_hat, _ = self.model.generate(self.config.eval.samples, self.device)
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
