import numpy as np
from absl import logging


def print_metrics(metrics, step: int | None = None):
    float_precision = np.get_printoptions()["precision"]
    log_strings = []

    if step is not None:
        log_strings.append(f"step={step}")
    for k, v in metrics.items():
        if isinstance(v, np.ndarray) and v.size == 1:
            v = v.item()
        if isinstance(v, float):
            log_strings.append(f"{k}={v:.{float_precision}f}")
        else:
            log_strings.append(f"{k}={v}")

    logging.info(" ".join(log_strings))
