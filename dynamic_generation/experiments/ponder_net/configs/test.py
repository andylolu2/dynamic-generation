from dynamic_generation.experiments.ponder_net.configs.base import get_ponder_net_config


def get_config():
    config = get_ponder_net_config()

    config.steps = -1

    train_conf = config.trainer_config
    train_conf.log_every = 1000
    train_conf.save_every = 5000
    train_conf.eval_every = 5000

    model_conf = train_conf.model
    model_conf.ponder_net_kwargs = dict(
        epsilon=0.05,
        lambda_p=0.7,
        beta=0.01,
        N_max=20,
    )

    data_conf = train_conf.dataset
    data_conf.dm_kwargs.dim = 64
    data_conf.train_kwargs.batch_size = 128
    data_conf.eval_kwargs.batch_size = 128
    data_conf.eval_kwargs.size = 25600

    return config
