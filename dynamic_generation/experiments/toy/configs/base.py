from ml_collections.config_dict import FieldReference

from dynamic_generation.utils.config import get_base_config


def get_config():
    config = get_base_config()
    config.project_name = "toy"

    std = FieldReference(0.02)

    config.log.every = 250
    config.save.every = -1
    config.eval.every = 1000
    config.eval.samples = 1000

    config.model_kwargs = dict(
        z_dim=1,
        hidden_dim=20,
        output_dim=2,
        z_samples=200,
        std=std,
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
