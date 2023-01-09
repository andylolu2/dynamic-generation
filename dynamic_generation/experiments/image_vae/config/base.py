from typing import Any

from ml_collections.config_dict import placeholder

from dynamic_generation.utils.config import get_base_config


def get_config() -> Any:
    config = get_base_config("image-vae")

    config.log.every = 200
    config.save.every = -1
    config.eval.every = 1000
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
        enc_dims=(512, 512),
        dec_dims=(512, 512),
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
            batch_size=256,
            size=-1,
        ),
        eval_kwargs=dict(
            batch_size=256,
            size=-1,
        ),
    )

    return config
