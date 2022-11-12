from ml_collections.config_dict import FieldReference, placeholder

from dynamic_generation.experiments.utils.config import get_base_config


def get_config():
    config = get_base_config("vae-lollipop")

    config.steps = -1
    config.restore = placeholder(str)
    config.trainer_config = dict(
        dry_run=config.get_ref("dry_run"),
        log_every=500,
        save_every=-1,
        eval_every=5000,
        save=dict(
            dir="model",
            ext=".pt",
        ),
        eval=dict(
            samples=2000,
        ),
        train=dict(
            beta_schedule_kwargs=dict(
                name="step",
                milestones=(1000, 15000),
                values=(0, 1e-2),
            ),
        ),
        model_kwargs=dict(
            z_dim=1,
            hidden_dim=64,
            input_dim=2,
            n_layers=4,
        ),
        optimizer_kwargs=dict(
            lr=3e-3,
        ),
        dataset=dict(
            dm_kwargs=dict(
                name="lollipop",
                std=0.02,
                arm_len=20,
                swirl_freq=0.75,
                n_rotations=1,
                swirl_prop=0.6,
            ),
            train_kwargs=dict(
                batch_size=128,
                size=10000,
            ),
            eval_kwargs=dict(
                batch_size=128,
                size=1280,
            ),
        ),
    )

    return config
