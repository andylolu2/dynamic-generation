from ml_collections.config_dict import FieldReference, placeholder

from dynamic_generation.experiments.utils.config import get_base_config


def get_ponder_net_config():
    config = get_base_config("ponder-net-repro")

    ds_dim = FieldReference(64)

    config.steps = -1
    config.restore = placeholder(str)
    config.trainer_config = dict(
        dry_run=config.get_ref("dry_run"),
        log_every=1000,
        save_every=-1,
        eval_every=5000,
        save=dict(
            dir="model",
            ext=".pt",
        ),
        train=dict(
            ponder_samples=10,
        ),
        eval=dict(
            ponder_samples=20,
        ),
        model=dict(
            ponder_module_kwargs=dict(
                input_size=ds_dim,
                hidden_size=128,
                output_size=1,
                num_layers=1,
            ),
            ponder_net_kwargs=dict(
                epsilon=0.05,
                lambda_p=0.7,
                beta=0.01,
                N_max=20,
            ),
        ),
        optimizer_kwargs=dict(
            name="Adam",
            lr=3e-4,
        ),
        dataset=dict(
            dm_kwargs=dict(
                name="parity",
                dim=ds_dim,
            ),
            train_kwargs=dict(
                batch_size=128,
                size=-1,
                min_n=1,
                max_n=ds_dim,
            ),
            eval_kwargs=dict(
                batch_size=128,
                size=12800,
                min_n=1,
                max_n=ds_dim,
            ),
        ),
    )

    return config
