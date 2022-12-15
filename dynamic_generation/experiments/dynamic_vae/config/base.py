from ml_collections import FieldReference

from dynamic_generation.experiments.utils.config import get_base_config


def get_config():
    config = get_base_config("dynamic-vae")

    std = FieldReference(0.02)

    config.log.every = 500
    config.save.every = -1
    config.eval.every = 2000
    config.eval.samples = 1000

    config.train = dict(
        beta_schedule_kwargs=dict(
            name="step",
            milestones=(0, 1000),
            values=(0, 1.0),
        ),
        grad_norm_clip=1.0,
    )

    config.model_kwargs = dict(
        z_dim=1,
        enc_dims=(20, 20, 20),
        dec_hidden_dim=20,
        dec_n_layers=2,
        epsilon=0.05,
        lambda_p=0.5,
        beta=0.1,
        N_max=20,
        std=std,
        average_halt_dist=False,
    )
    config.optimizer_kwargs = dict(
        name="Adam",
        lr=3e-3,
    )
    config.dataset = dict(
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
            batch_size=128,
            size=1280,
        ),
    )

    return config
