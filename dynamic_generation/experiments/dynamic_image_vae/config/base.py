from typing import Any

from ml_collections.config_dict import placeholder

from dynamic_generation.utils.config import get_base_config


def get_config() -> Any:
    config = get_base_config()
    config.project_name = "dynamic-image-vae"

    config.log.every = 500
    config.save.every = -1
    config.eval.every = 2000
    config.eval.generate_samples = 16
    config.eval.reconstruct_samples = 5

    config.train = dict(
        grad_norm_clip=placeholder(float),
        beta_schedule_kwargs=dict(
            name="constant",
            value=1.0,
        ),
    )
    config.model_kwargs = dict(
        z_dim=2,
        enc_dims=(64, 64, 64, 64),
        ponder_module_kwargs=dict(
            hidden_size=128,
            num_layers=1,
        ),
        ponder_net_kwargs=dict(
            epsilon=0.05,
            lambda_p=0.5,
            beta=0.01,
            N_max=10,
            average_halt_dist=False,
        ),
    )
    config.optimizer_kwargs = dict(
        name="Adam",
        lr=1e-3,
    )
    config.dataset = dict(
        dm_kwargs=dict(
            name="mnist",
            data_path="./data",
            size=28,
            range_=(0, 1),
        ),
        train_kwargs=dict(
            batch_size=64,
            size=-1,
        ),
        eval_kwargs=dict(
            batch_size=256,
            size=-1,
        ),
    )

    return config
