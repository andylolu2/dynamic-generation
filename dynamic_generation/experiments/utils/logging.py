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
            v_str = str(v)
            if len(v_str) < 20 and not (v_str.startswith("<") and v_str.endswith(">")):
                log_strings.append(f"{k}={v_str}")

    logging.info(" ".join(log_strings))
