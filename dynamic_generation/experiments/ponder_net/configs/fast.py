from dynamic_generation.experiments.ponder_net.configs.base import get_ponder_net_config


def get_config():
    config = get_ponder_net_config()

    config.steps = 30000

    config.log.every = 500
    config.save.every = 2000
    config.eval.every = 2000

    config.model.ponder_net_kwargs = dict(
        epsilon=0.05,
        lambda_p=0.2,
        beta=0.01,
        N_max=20,
    )
    config.model.ponder_module_kwargs.hidden_size = 64

    data_conf = config.dataset
    data_conf.dm_kwargs.dim = 8
    data_conf.train_kwargs.batch_size = 128
    data_conf.eval_kwargs.batch_size = 128
    data_conf.eval_kwargs.size = 12800

    return config
