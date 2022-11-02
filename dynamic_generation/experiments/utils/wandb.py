from contextlib import contextmanager
from typing import Sequence

import wandb


@contextmanager
def wandb_run(
    project: str | None = None,
    dry_run: bool = False,
    config: dict | None = None,
    tags: Sequence[str] | None = None,
):
    if not dry_run:
        wandb.login()

    mode = "disabled" if dry_run else "online"
    wandb.init(mode=mode, project=project, config=config, tags=tags)

    try:
        yield
    finally:
        wandb.finish()
