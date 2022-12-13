from dynamic_generation.experiments.ponder_net.configs.base import get_ponder_net_config


def get_config():
    config = get_ponder_net_config()

    config.log.every = 500
    config.save.every = 2000
    config.eval.every = 2000

    config.model.ponder_net_kwargs = dict(
        epsilon=0.05,
        lambda_p=0.2,
        beta=0.01,
        N_max=20,
    )

    config.dataset.dm_kwargs.dim = 96
    config.dataset.train_kwargs.max_n = 48
    config.dataset.eval_kwargs.min_n = 49
    config.dataset.eval_kwargs.max_n = 96

    return config
