from dynamic_generation.types import Tensor

EPSILON = 1e-7


def safe_log(x: Tensor):
    return (x + EPSILON).log()
