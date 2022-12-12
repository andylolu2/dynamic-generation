from dynamic_generation.experiments.ponder_net.configs.base import get_ponder_net_config


def get_config():
    config = get_ponder_net_config()

    config.steps = 100
    config.log.every = 5
    config.eval.every = 10

    config.model.ponder_net_kwargs = dict(
        epsilon=0.05,
        lambda_p=0.1,
        beta=0.01,
        N_max=10,
    )

    data_conf = config.dataset
    data_conf.dm_kwargs.dim = 8
    data_conf.train_kwargs.batch_size = 8
    data_conf.eval_kwargs.batch_size = 4
    data_conf.eval_kwargs.size = 8

    return config
