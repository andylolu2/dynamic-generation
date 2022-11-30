import torch


def load_optimizer(name: str, *args, **kwargs):
    optimizer_cls = getattr(torch.optim, name)
    return optimizer_cls(*args, **kwargs)
