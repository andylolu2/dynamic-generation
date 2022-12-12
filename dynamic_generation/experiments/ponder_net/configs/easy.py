from dynamic_generation.experiments.ponder_net.configs.base import get_ponder_net_config


def get_config():
    config = get_ponder_net_config()

    config.log.every = 1000
    config.eval.every = 2000

    config.model.ponder_net_kwargs = dict(
        epsilon=0.05,
        lambda_p=0.5,
        beta=0.1,
        N_max=20,
    )

    data_conf = config.dataset
    data_conf.dm_kwargs.dim = 16
    data_conf.train_kwargs.batch_size = 128
    data_conf.eval_kwargs.batch_size = 128
    data_conf.eval_kwargs.size = 12800

    return config
