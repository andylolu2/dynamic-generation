import torch
import wandb
from absl import app, logging
from torch.cuda.amp.grad_scaler import GradScaler

from dynamic_generation.experiments.trainer import Trainer
from dynamic_generation.models.mlp_diffusion import MLPDiffuser
from dynamic_generation.models.unet_diffusion import UNetDiffuser
from dynamic_generation.types import TrainState
from dynamic_generation.utils.figure import new_figure
from dynamic_generation.utils.image import tile_image
from dynamic_generation.utils.metrics import global_metrics
from dynamic_generation.utils.optimizers import load_optimizer
from dynamic_generation.utils.torch import count_parameters


class DiffusionTrainer(Trainer):
    def initialize_state(self) -> TrainState:
        state = super().initialize_state()

        match self.config.model.name:
            case "mlp":
                model_cls = MLPDiffuser
            case "unet":
                model_cls = UNetDiffuser
            case x:
                raise ValueError(f"Unsupported model: {x}")

        self.model = model_cls(
            input_dims=self.data_module.shape["x"], **self.config.model.kwargs
        )

        self.model.to(self.device)
        self.optimizer = load_optimizer(
            params=self.model.parameters(), **self.config.optimizer_kwargs
        )
        self.scaler = GradScaler(enabled=(self.config.precision == "mixed"))

        state["model"] = self.model
        state["optimizer"] = self.optimizer
        state["scaler"] = self.scaler

        logging.info(f"No. parameters: {count_parameters(self.model) / 1e6:.3f}M")

        return state

    def _step(self, item):
        x = self.cast(item["x"])

        # forward pass
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            loss = self.model.loss(x)

        # backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()  # type: ignore
        self.scaler.unscale_(self.optimizer)
        self.clip_grad(self.model.parameters(), self.config.train.grad_norm_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()

        for item in self.eval_loader:
            x = self.cast(item["x"])
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                _ = self.model.loss(x)

        generate_conf = self.config.eval.generate
        x_T = torch.randn(
            generate_conf.n_samples, *self.data_module.shape["x"], device=self.device
        )
        x_0 = self.model.generate(x_T, generate_conf.steps, mode=generate_conf.mode)
        img = tile_image(x_0, input_range=(-1, 1))

        with new_figure(show=self.config.dry_run) as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img)
            ax.grid(False)
            fig.tight_layout()
            global_metrics.log("samples", wandb.Image(fig))

        self.model.train()


if __name__ == "__main__":
    app.run(DiffusionTrainer.run)
