from dynamic_generation.experiments.ponder_net.configs.base import get_ponder_net_config


def get_config():
    config = get_ponder_net_config()

    config.steps = 100
    config.log.every = 5
    config.eval.every = 10

    config.model.ponder_net_kwargs.N_max = 10

    config.dataset.dm_kwargs.dim = 8
    config.dataset.train_kwargs.max_n = 8
    config.dataset.eval_kwargs.max_n = 8

    config.dataset.train_kwargs.batch_size = 8
    config.dataset.eval_kwargs.batch_size = 8
    config.dataset.eval_kwargs.size = 32

    return config
