import numpy as np
import torch
import torch.distributions as D
import wandb
from absl import app
from matplotlib import colors
from ml_collections import FrozenConfigDict

from dynamic_generation.experiments.trainer import Trainer
from dynamic_generation.experiments.vae.train import VAETrainer
from dynamic_generation.models.ponder_module import RNNPonderModule
from dynamic_generation.models.ponder_net import PonderNet
from dynamic_generation.types import TrainState
from dynamic_generation.utils.distributions import CustomMixture
from dynamic_generation.utils.figure import new_figure
from dynamic_generation.utils.metrics import global_metrics
from dynamic_generation.utils.optimizers import load_optimizer
from dynamic_generation.utils.secrets import secrets
from dynamic_generation.utils.wandb import load_runs


class GuidedDynamicVAETrainer(Trainer):
    def initialize_state(self) -> TrainState:
        train_state = super().initialize_state()

        ponder_module = RNNPonderModule(**self.config.model.ponder_module_kwargs)
        model = PonderNet(
            ponder_module=ponder_module, **self.config.model.ponder_net_kwargs
        )
        model.to(device=self.device, dtype=self.dtype)

        optimizer = load_optimizer(
            params=model.parameters(), **self.config.optimizer_kwargs
        )

        train_state["model"] = model
        train_state["optimizer"] = optimizer

        self.model = model
        self.optimizer = optimizer

        run = load_runs(
            secrets.wandb.entity,
            self.config.teacher.project,
            self.config.teacher.run_id,
        )
        trainer = VAETrainer(FrozenConfigDict(run.config), mode="inference")
        trainer.load(self.config.teacher.checkpoint)
        self.teacher = trainer.model
        self.z_dist = self.teacher.prior

        return train_state

    def _step(self, _):
        z = self.z_dist.sample((self.config.train.batch_size,)).to(self.device)
        with torch.inference_mode(), global_metrics.capture(None):
            X_target, _ = self.teacher.decode(z)

        x_hats, halt_dist = self.model(z)
        X_hats = D.Independent(D.Normal(loc=x_hats, scale=self.teacher.std), 1)
        X_hat = CustomMixture(halt_dist, X_hats)
        loss = self.model.loss(y_true=X_target.mean, y_pred=X_hat, halt_dist=halt_dist)

        self.optimizer.zero_grad()
        loss.backward()
        self.clip_grad(self.model.parameters(), self.config.train.grad_norm_clip)
        self.optimizer.step()

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()

        z = self.z_dist.sample((self.config.eval.samples,)).to(self.device)
        x_hats, halt_dist = self.model.forward(z)
        X_hats = D.Independent(D.Normal(loc=x_hats, scale=self.teacher.std), 1)
        X_hat = CustomMixture(halt_dist, X_hats)

        # generate samples & plot
        x_hat, mix_sample = X_hat.sample_detailed(mode="deterministic")
        x_hat = x_hat.cpu().numpy()
        mix_sample = mix_sample.cpu().numpy()

        with new_figure(show=self.config.dry_run) as fig:
            norm = colors.Normalize(mix_sample.min(), mix_sample.max())
            ax = fig.add_subplot(1, 1, 1)
            path = ax.scatter(
                x=x_hat[:, 0],
                y=x_hat[:, 1],
                s=2,
                norm=norm,
                c=mix_sample[:, 0],
                cmap="rainbow",
                label="Generated",
            )
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(path)
            global_metrics.log("samples", wandb.Image(fig))

        # plot halt distribution
        average_dist = halt_dist.average().probs.cpu().numpy()
        dummy = 1 + np.arange(halt_dist.N)

        with new_figure(show=self.config.dry_run) as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(
                dummy,
                bins=halt_dist.N - 1,
                weights=average_dist,
                density=True,
            )
            global_metrics.log("halt_dist", wandb.Image(fig))

        self.model.train()


if __name__ == "__main__":
    app.run(GuidedDynamicVAETrainer.run)
