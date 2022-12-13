from typing import Any

from ml_collections.config_dict import FieldReference, placeholder

from dynamic_generation.experiments.utils.config import get_base_config


def get_ponder_net_config() -> Any:
    config = get_base_config("ponder-net-repro")

    ds_dim = FieldReference(64)

    config.train = dict(
        grad_norm_clip=placeholder(float),
    )
    config.model = dict(
        ponder_module_kwargs=dict(
            input_size=ds_dim,
            hidden_size=128,
            output_size=1,
            num_layers=1,
        ),
        ponder_net_kwargs=dict(
            epsilon=0.05,
            lambda_p=0.2,
            beta=0.01,
            N_max=20,
        ),
    )
    config.optimizer_kwargs = dict(
        name="Adam",
        lr=3e-4,
    )
    config.dataset = dict(
        dm_kwargs=dict(
            name="parity",
            dim=ds_dim,
        ),
        train_kwargs=dict(
            batch_size=128,
            size=-1,
            min_n=1,
            max_n=ds_dim.get(),
        ),
        eval_kwargs=dict(
            batch_size=128,
            size=12800,
            min_n=1,
            max_n=ds_dim.get(),
        ),
    )

    return config
