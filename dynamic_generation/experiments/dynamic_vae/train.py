import numpy as np
import torch
import wandb
from absl import app
from matplotlib import colors

from dynamic_generation.experiments.trainer import Trainer
from dynamic_generation.models.vae import DynamicVae
from dynamic_generation.types import TrainState
from dynamic_generation.utils.figure import new_figure
from dynamic_generation.utils.metrics import global_metrics
from dynamic_generation.utils.optimizers import load_optimizer
from dynamic_generation.utils.schedules import load_schedule


class DynamicVAETrainer(Trainer):
    def initialize_state(self) -> TrainState:
        train_state = super().initialize_state()

        x_shape = self.data_module.shape["x"]
        assert len(x_shape) == 1

        model = DynamicVae(input_dim=x_shape[0], **self.config.model_kwargs)
        optimizer = load_optimizer(
            params=model.parameters(), **self.config.optimizer_kwargs
        )

        model.to(device=self.device, dtype=self.dtype)

        train_state["model"] = model
        train_state["optimizer"] = optimizer

        self.beta_schedule = load_schedule(**self.config.train.beta_schedule_kwargs)
        self.model = model
        self.optimizer = optimizer

        return train_state

    def _step(self, item):
        x = self.cast(item["x"])
        out = self.model(x)

        beta = self.beta_schedule(self.train_step)
        loss = self.model.loss(x, out, beta)

        self.optimizer.zero_grad()
        loss.backward()
        self.clip_grad(self.model.parameters(), self.config.train.grad_norm_clip)
        self.optimizer.step()

        global_metrics.log("beta", beta, "mean")

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

        # generate samples & plot
        x_hat, aux = self.model.generate(self.config.eval.samples, self.device)
        data = x_hat.cpu().numpy()
        halt_step = aux["mix_sample"][:, 0].cpu().numpy() + 1
        norm = colors.Normalize(halt_step.min(), halt_step.max())

        with new_figure(show=self.config.dry_run) as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(x=real[:, 0], y=real[:, 1], s=1, c="k", zorder=1, label="Real")
            path = ax.scatter(
                x=data[:, 0],
                y=data[:, 1],
                s=2,
                zorder=2,
                norm=norm,
                c=halt_step,
                cmap="rainbow",
                label="Generated",
            )
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(path)

            global_metrics.log("samples", wandb.Image(fig), "replace")

        # plot halt distribution
        halt_dist = aux["halt_dist"]
        average_dist = halt_dist.average().probs.cpu().numpy()
        dummy = 1 + np.arange(halt_dist.N)

        with new_figure(show=self.config.dry_run) as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(
                dummy,
                bins=halt_dist.N - 1,
                weights=average_dist,
                density=True,
                edgecolor="w",
                color="skyblue",
            )

            global_metrics.log("halt_dist", wandb.Image(fig), "replace")

        self.model.train()


if __name__ == "__main__":
    app.run(DynamicVAETrainer.run)
