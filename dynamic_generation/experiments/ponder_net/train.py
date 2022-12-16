import numpy as np
import seaborn as sns
import torch
import torch.distributions as D
import wandb
from absl import app, logging
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torchmetrics.functional.classification import binary_accuracy

from dynamic_generation.experiments.trainer import Trainer
from dynamic_generation.models.ponder_module import GRUPonderModule, RNNPonderModule
from dynamic_generation.models.ponder_net import PonderNet
from dynamic_generation.types import TrainState
from dynamic_generation.utils.accumulators import StackAccumulator
from dynamic_generation.utils.distributions import CustomMixture
from dynamic_generation.utils.figure import new_figure
from dynamic_generation.utils.metrics import global_metrics
from dynamic_generation.utils.optimizers import load_optimizer


class PonderNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        x_shape = self.data_module.shape["x"]
        assert len(x_shape) == 1
        self.ds_dim = x_shape[0]

        self.model: PonderNet = self.train_state["model"]
        self.optimizer: Optimizer = self.train_state["optimizer"]
        self.scaler: GradScaler = self.train_state["scaler"]

    def initialize_state(self) -> TrainState:
        state = super().initialize_state()

        # ponder_module = GRUPonderModule(**self.config.model.ponder_module_kwargs)
        ponder_module = RNNPonderModule(
            input_size=self.ds_dim, **self.config.model.ponder_module_kwargs
        )
        model = PonderNet(
            ponder_module=ponder_module, **self.config.model.ponder_net_kwargs
        )
        optimizer = load_optimizer(
            params=model.parameters(), **self.config.optimizer_kwargs
        )
        scaler = GradScaler(enabled=(self.precision == "mixed"))

        model.to(device=self.device)

        state["model"] = model
        state["optimizer"] = optimizer
        state["scaler"] = scaler
        return state

    def _step(self, item):
            x, y_true = item["x"], item["y"]
            x, y_true = self.cast(x, y_true)

            # forward pass
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                y, halt_dist = self.model(x)
                y_pred = CustomMixture(
                    mixture_distribution=halt_dist,
                    component_distribution=D.Bernoulli(logits=y.squeeze(-1)),
                )
            loss = self.model.loss(y_true=y_true, y_pred=y_pred, halt_dist=halt_dist)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.clip_grad(self.model.parameters(), self.config.train.grad_norm_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            pred = y_pred.sample()
            target, pred = torch.broadcast_tensors(y_true, pred)
            accuracy = binary_accuracy(pred, target)

            global_metrics.log("accuracy", accuracy.item(), "mean")

    @torch.inference_mode()
    def evaluate(self):
        logging.info("Begin evaluation...")

        self.model.eval()

        num_ones = StackAccumulator(batched=True)
        num_steps = StackAccumulator(batched=True)

        for item in self.eval_loader:
            x, y_true = item["x"], item["y"]
            x, y_true = self.cast(x, y_true)
            num_ones.update(x.abs().sum(-1))

            y, halt_dist = self.model(x)
            y_pred = CustomMixture(
                mixture_distribution=halt_dist,
                component_distribution=D.Bernoulli(logits=y.squeeze(-1)),
            )
            _ = self.model.loss(y_true=y_true, y_pred=y_pred, halt_dist=halt_dist)

            # calculate weighted accuracy
            pred, mix_sample = y_pred.sample_detailed()
            num_steps.update(mix_sample)
            target, pred = torch.broadcast_tensors(y_true, pred)
            accuracy = binary_accuracy(pred, target)
            global_metrics.log("accuracy", accuracy.item(), "mean")

        num_ones = num_ones.compute().astype(np.int64)
        num_steps = num_steps.compute().astype(np.int64) + 1

        with new_figure(show=self.config.dry_run) as fig:
            ax = fig.add_subplot(1, 1, 1)
            sns.histplot(
                x=num_ones,
                y=num_steps,
                binrange=[
                    (0, self.ds_dim),
                    (0, self.config.model.ponder_net_kwargs.N_max),
                ],
                discrete=True,
                ax=ax,
            )
            global_metrics.log("halt_dist_by_num_ones", wandb.Image(fig), "replace")

        self.model.train()


if __name__ == "__main__":
    app.run(PonderNetTrainer.run)
