from torch import nn


def load_activation(name: str) -> nn.Module:
    return getattr(nn, name)()
