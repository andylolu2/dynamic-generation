from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_base_config(project_name: str):
    config = ConfigDict()

    config.project_name = project_name
    config.tags = placeholder(tuple)
    config.notes = placeholder(str)
    config.logging = dict(
        level=logging.INFO,
        format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
        time_format="%Y-%m-%d %H:%M:%S",
        float_precision=4,
    )
    config.dry_run = False
    config.restore = placeholder(str)
    config.precision = "full"

    return config
