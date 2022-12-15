from dynamic_generation.experiments.ponder_net.configs.base import get_ponder_net_config


def get_config():
    config = get_ponder_net_config()

    config.log.every = 500
    config.save.every = 2000
    config.eval.every = 2000

    config.model.ponder_net_kwargs.lambda_p = 0.2
    config.model.ponder_module_kwargs.hidden_size = 64

    config.dataset.dm_kwargs.dim = 8
    config.dataset.train_kwargs.max_n = 8
    config.dataset.eval_kwargs.max_n = 8

    return config
