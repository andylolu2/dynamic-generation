from typing import Any

from dynamic_generation.utils.config import get_base_config


def get_config() -> Any:
    config = get_base_config()
    config.project_name = "mlp-diffusion"

    config.log.every = 500
    config.save.every = -1
    config.eval.every = 5000
    config.eval.generate = dict(
        n_samples=16,
        steps=100,
        mode="ddim",
    )

    config.train = dict(grad_norm_clip=1.0)
    config.model = dict(
        name="mlp",
        kwargs=dict(
            dims=(1024,),
            noise_schedule_kwargs=dict(
                beta_1=1e-4,
                beta_T=0.02,
                steps=1000,
                scale=4,
            ),
            time_embedding_kwargs=dict(
                dim=64,
                max_period=10000,
            ),
        ),
    )
    config.optimizer_kwargs = dict(
        name="Adam",
        lr=1e-4,
    )
    config.dataset = dict(
        dm_kwargs=dict(
            name="mnist",
            data_path="./data",
            size=32,
            range_=(-1, 1),
        ),
        train_kwargs=dict(
            batch_size=64,
            size=-1,
        ),
        eval_kwargs=dict(
            batch_size=64,
            size=-1,
        ),
    )

    return config
