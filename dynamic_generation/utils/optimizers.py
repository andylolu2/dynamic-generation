import torch


def load_optimizer(name: str, *args, **kwargs) -> torch.optim.Optimizer:
    optimizer_cls = getattr(torch.optim, name)
    return optimizer_cls(*args, **kwargs)
