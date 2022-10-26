from ml_collections.config_dict import placeholder

from dynamic_generation.experiments.utils.config import get_base_config


def get_config():
    config = get_base_config("ponder-net-repro")

    ds_dim = 64

    config.steps = -1
    config.restore = placeholder(str)
    config.trainer_config = dict(
        dry_run=config.get_ref("dry_run"),
        log_every=1000,
        save_every=5000,
        eval_every=5000,
        save=dict(
            dir="model",
            ext=".pt",
        ),
        model=dict(
            ponder_module_kwargs=dict(
                input_size=ds_dim,
                hidden_size=128,
                output_size=1,
            ),
            ponder_net_kwargs=dict(
                epsilon=0.05,
                lambda_p=0.6,
                beta=0.01,
                N_max=25,
            ),
        ),
        optimizer_kwargs=dict(
            lr=3e-4,
        ),
        dataset_kwargs=dict(
            ds_kwargs=dict(
                dim=ds_dim,
            ),
            train_kwargs=dict(
                batch_size=128,
                size=-1,
            ),
            eval_kwargs=dict(
                batch_size=128,
                size=25600,
            ),
        ),
    )

    return config
