from ml_collections.config_dict import FieldReference, placeholder

from dynamic_generation.experiments.utils.config import get_base_config


def get_config():
    config = get_base_config("static-lollipop")

    std = FieldReference(0.03)
    config.steps = -1
    config.restore = placeholder(str)
    config.trainer_config = dict(
        dry_run=config.get_ref("dry_run"),
        log_every=100,
        save_every=-1,
        eval_every=500,
        save=dict(
            dir="model",
            ext=".pt",
        ),
        eval=dict(
            samples=1000,
        ),
        model_kwargs=dict(
            z_dim=1,
            hidden_dim=20,
            output_dim=2,
            z_samples=200,
            std=std,
        ),
        optimizer_kwargs=dict(
            lr=1e-3,
        ),
        dataset=dict(
            dm_kwargs=dict(
                name="lollipop",
                std=std,
                arm_len=20,
                swirl_freq=1,
                n_rotations=1,
                swirl_prop=0.5,
            ),
            train_kwargs=dict(
                batch_size=128,
                size=10000,
            ),
            eval_kwargs=dict(
                batch_size=64,
                size=1280,
            ),
        ),
    )

    return config
