import torch
import wandb
from absl import app, logging
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from dynamic_generation.experiments.trainer import Trainer
from dynamic_generation.models.image_vae import ImageVae
from dynamic_generation.types import TrainState
from dynamic_generation.utils.figure import new_figure
from dynamic_generation.utils.image import tile_image
from dynamic_generation.utils.metrics import global_metrics
from dynamic_generation.utils.optimizers import load_optimizer
from dynamic_generation.utils.schedules import load_schedule


class StaticImageVaeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta_schedule = load_schedule(**self.config.train.beta_schedule_kwargs)
        self.model: ImageVae = self.train_state["model"]
        self.optimizer: Optimizer = self.train_state["optimizer"]
        self.scaler: GradScaler = self.train_state["scaler"]

    def initialize_state(self) -> TrainState:
        state = super().initialize_state()
        model = ImageVae(
            input_dims=self.data_module.shape["x"], **self.config.model_kwargs
        )
        optimizer = load_optimizer(
            params=model.parameters(), **self.config.optimizer_kwargs
        )

        model.to(device=self.device)

        state["model"] = model
        state["optimizer"] = optimizer
        scaler = GradScaler(enabled=(self.precision == "mixed"))
        state["scaler"] = scaler
        return state

    def _step(self, item):
        with global_metrics.capture("train"):
            x = self.cast(item["x"])

            # forward pass
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                out = self.model(x)
                beta = self.beta_schedule(self.train_step)
                loss = self.model.loss(x, out, beta)

            # backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # compute and clip grad norm
            self.scaler.unscale_(self.optimizer)
            if self.config.train.grad_norm_clip is not None:
                grad_norm = nn.utils.clip_grad.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_norm_clip
                )
                global_metrics.log("grad_norm", grad_norm.item(), "mean")

            # update parameters
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if "epoch" in item:
                global_metrics.log("epoch", item["epoch"])

    @torch.inference_mode()
    def evaluate(self):
        logging.info("Begin evaluation...")

        self.model.eval()
        reconstruction = None

        for item in self.eval_loader:
            x = self.cast(item["x"])

            # forward pass
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                out = self.model(x)
                _ = self.model.loss(x, out)

            if reconstruction is None:
                reconstruction = (
                    x[: self.config.eval.reconstruct_samples],
                    out["X_hat"].base_dist.probs[
                        : self.config.eval.reconstruct_samples
                    ],
                )

        with new_figure(show=self.config.dry_run) as fig:
            ax = fig.add_subplot(2, 1, 1)
            img = tile_image(reconstruction[0], nrow=1)
            ax.imshow(img)
            ax.grid(False)
            ax.set_title("Original")

            ax = fig.add_subplot(2, 1, 2)
            img = tile_image(reconstruction[1], nrow=1)
            ax.imshow(img)
            ax.grid(False)
            ax.set_title("Reconstructed")

            fig.tight_layout()
            global_metrics.log("reconstruction", wandb.Image(fig))

        x_hat, _ = self.model.generate(self.config.eval.generate_samples, self.device)
        img = tile_image(x_hat)

        with new_figure(show=self.config.dry_run) as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img)
            ax.grid(False)

            fig.tight_layout()
            global_metrics.log("samples", wandb.Image(fig))

        self.model.train()


if __name__ == "__main__":
    app.run(StaticImageVaeTrainer.run)
