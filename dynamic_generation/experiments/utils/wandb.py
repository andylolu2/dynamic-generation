from contextlib import contextmanager

import wandb


@contextmanager
def wandb_run(
    project: str | None = None,
    dry_run: bool = False,
    config: dict | None = None,
):
    if not dry_run:
        wandb.login()

    mode = "disabled" if dry_run else "online"
    wandb.init(mode=mode, project=project, config=config)

    try:
        yield
    finally:
        wandb.finish()
