from typing import Any

from ml_collections import FieldReference

from dynamic_generation.utils.config import get_base_config


def get_config() -> Any:
    config = get_base_config()
    config.project_name = "unet-diffusion"

    config.log.every = 500
    config.save.every = 5000
    config.eval.every = 5000
    config.eval.generate = dict(
        n_samples=16,
        steps=100,
        mode="ddim",
    )

    config.train = dict(grad_norm_clip=1.0)

    dim_base = FieldReference(48)
    attention_heads = FieldReference(4)

    config.model = dict(
        name="unet",
        kwargs=dict(
            noise_schedule_kwargs=dict(
                beta_1=1e-4,
                beta_T=0.02,
                steps=1000,
                scale=1,
            ),
            unet_kwargs=dict(
                dim_base=dim_base,
                dim_time=4 * dim_base,
                blocks_per_level=2,
                dim_mults=(1, 2, 2, 2),
                attention_resolutions=(4,),
                time_emb_kwargs=dict(
                    dim=dim_base,
                    max_period=10000,
                ),
                resnet_block_kwargs=dict(
                    groups=4,
                ),
                self_attention_kwargs=dict(
                    heads=attention_heads,
                    dim_head=dim_base // attention_heads,
                ),
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
