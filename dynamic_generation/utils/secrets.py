from typing import Any

import yaml
from ml_collections import FrozenConfigDict

with open("secrets.yaml", "r") as f:
    secrets: Any = FrozenConfigDict(yaml.safe_load(f))
