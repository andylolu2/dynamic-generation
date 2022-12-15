from contextlib import contextmanager

import wandb


@contextmanager
def wandb_run(dry_run: bool = False, *args, **kwargs):
    if not dry_run:
        wandb.login()

    mode = "disabled" if dry_run else "online"
    wandb.init(mode=mode, dir="wandb_logs", *args, **kwargs)

    try:
        yield
    finally:
        wandb.finish()
