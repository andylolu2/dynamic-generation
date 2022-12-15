from pathlib import Path

import torch
import wandb
from absl import app
from torch import nn
from torch.optim import Optimizer

from dynamic_generation.experiments.trainer import Trainer
from dynamic_generation.experiments.utils.metrics import global_metrics
from dynamic_generation.experiments.utils.optimizers import load_optimizer
from dynamic_generation.experiments.utils.schedules import load_schedule
from dynamic_generation.models.vae import UniformBetaVAE
from dynamic_generation.types import TrainState
from dynamic_generation.utils.figure import new_figure


class VAETrainer(Trainer):
    def __init__(self, config, exp_dir: Path):
        super().__init__(config, exp_dir)

        self.beta_schedule = load_schedule(**self.config.train.beta_schedule_kwargs)
        self.model: UniformBetaVAE = self.train_state["model"]
        self.optimizer: Optimizer = self.train_state["optimizer"]

    def initialize_state(self) -> TrainState:
        train_state = super().initialize_state()
        model = UniformBetaVAE(**self.config.model_kwargs)
        optimizer = load_optimizer(
            params=model.parameters(), **self.config.optimizer_kwargs
        )

        model.to(device=self.device, dtype=self.dtype)

        train_state["model"] = model
        train_state["optimizer"] = optimizer
        return train_state

    def _step(self, item):
        with global_metrics.capture("train"):
            x = self.cast(item["x"])
            out = self.model(x)

            beta = self.beta_schedule(self.train_step)
            loss = self.model.loss(x, out, beta)

            self.optimizer.zero_grad()
            loss.backward()

            if self.config.train.grad_norm_clip is not None:
                grad_norm = nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_norm_clip
                )
                global_metrics.log("grad_norm", grad_norm.item(), "mean")
            self.optimizer.step()

            if "epoch" in item:
                global_metrics.log("epoch", item["epoch"], "replace")

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()

        xs = []
        for item in self.eval_loader:
            x = self.cast(item["x"])
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
        with new_figure(show=self.config.dry_run) as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(x=real[:, 0], y=real[:, 1], s=1, c="r", zorder=1, label="Real")
            ax.scatter(
                x=data[:, 0], y=data[:, 1], s=1, c="b", zorder=2, label="Generated"
            )
            ax.set(box_aspect=1)
            fig.tight_layout()

            global_metrics.log("samples", wandb.Image(fig), "replace")

        self.model.train()


if __name__ == "__main__":
    app.run(VAETrainer.run)