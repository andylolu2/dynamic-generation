from typing import Any

from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_base_config(project_name: str) -> Any:
    config = ConfigDict()

    config.project_name = project_name
    config.tags = placeholder(tuple)
    config.notes = placeholder(str)
    config.log = dict(
        every=1000,
        level=logging.INFO,
        format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
        time_format="%Y-%m-%d %H:%M:%S",
        float_precision=4,
    )
    config.dry_run = False
    config.steps = -1
    config.restore = placeholder(str)
    config.precision = "full"

    config.save = dict(
        every=-1,
        dir="model",
        ext=".pt",
    )
    config.eval = dict(
        every=-1,
    )
    config.benchmark = dict(
        run=False,
        warmup_steps=50,
        steps=100,
    )

    return config
