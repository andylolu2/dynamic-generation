from ml_collections.config_dict import placeholder

from dynamic_generation.utils.config import get_base_config


def get_config():
    config = get_base_config("guided-dynamic-vae")

    config.log.every = 500
    config.save.every = 2000
    config.eval.every = 2000
    config.eval.samples = 1000

    config.teacher = dict(
        project="vae",
        run_id=placeholder(int),
        checkpoint=placeholder(str),
    )

    config.train = dict(
        grad_norm_clip=None,
        batch_size=128,
    )

    config.model = dict(
        ponder_module_kwargs=dict(
            input_size=1,
            hidden_size=40,
            output_size=2,
            num_layers=1,
        ),
        ponder_net_kwargs=dict(
            epsilon=0.05,
            lambda_p=0.5,
            beta=0.6,
            N_max=20,
            average_halt_dist=False,
        ),
    )
    config.optimizer_kwargs = dict(
        name="Adam",
        lr=1e-3,
    )
    config.dataset = dict(
        dm_kwargs=dict(name="empty"),
        train_kwargs=dict(),
        eval_kwargs=dict(),
    )

    return config
