import matplotlib.pyplot as plt
import torch
import wandb
from absl import app
from torch.optim import Optimizer

from dynamic_generation.experiments.trainer import Trainer
from dynamic_generation.experiments.utils.metrics import global_metrics
from dynamic_generation.experiments.utils.optimizers import load_optimizer
from dynamic_generation.models.toy_generator import ToyGenerator
from dynamic_generation.types import TrainState


class ToyGeneratorTrainer(Trainer):
    @property
    def model(self) -> ToyGenerator:
        return self.train_state["model"]

    @property
    def optimizer(self) -> Optimizer:
        return self.train_state["optimizer"]

    def initialize_state(self) -> TrainState:
        train_state = super().initialize_state()
        model = ToyGenerator(trainer=self, **self.config.model_kwargs)
        optimizer = load_optimizer(
            params=model.parameters(), **self.config.optimizer_kwargs
        )

        model.to(device=self.device, dtype=self.dtype)

        train_state["model"] = model
        train_state["optimizer"] = optimizer
        return train_state

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


if __name__ == "__main__":
    app.run(ToyGeneratorTrainer.run)
