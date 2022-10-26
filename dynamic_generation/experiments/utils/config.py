from pathlib import Path

from absl import logging
from ml_collections import ConfigDict, FieldReference


def get_base_config(project_name: str):
    config = ConfigDict()

    config.project_name = project_name
    config.logging = dict(
        level=logging.INFO,
        format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
        time_format="%Y-%m-%d %H:%M:%S",
        float_precision=4,
    )
    config.dry_run = False

    return config
