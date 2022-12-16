from contextlib import contextmanager
from typing import overload

import wandb
from wandb.wandb_run import Run


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


@overload
def load_runs(entity: str, project: str, run_ids: int) -> Run:
    ...


@overload
def load_runs(entity: str, project: str, run_ids: list[int]) -> list[Run]:
    ...


def load_runs(entity: str, project: str, run_ids: list[int] | int):
    api = wandb.Api()
    singular = isinstance(run_ids, int)

    if singular:
        run_ids = [run_ids]

    runs = api.runs(
        path=f"{entity}/{project}",
        filters={
            "display_name": {
                "$regex": rf"^[a-zA-Z]+-[a-zA-Z]+-({'|'.join(map(str, run_ids))})$"
            }
        },
    )

    if len(runs) != len(run_ids):
        run_names = [run.name for run in runs]
        raise ValueError(f"Request: {run_ids}, response: {run_names}")

    if singular:
        return runs[0]
    else:
        return runs
