import difflib

import wandb
from absl import app, flags, logging
from colorama import Fore, Style
from ml_collections import ConfigDict

FLAGS = flags.FLAGS

flags.DEFINE_string("entity", None, "The username / entity.")
flags.DEFINE_string("project", None, "The project name.")
flags.DEFINE_string("run_1", None, "The run number of the first run.")
flags.DEFINE_string("run_2", None, "The run number of the second run.")

flags.mark_flags_as_required(["entity", "project", "run_1", "run_2"])


def to_config_str(run):
    config = ConfigDict(run.config)
    return str(config).splitlines(keepends=True)


def main(argv):
    logging.level_info()

    api = wandb.Api()
    runs = api.runs(
        path=f"{FLAGS.entity}/{FLAGS.project}",
        filters={"display_name": {"$regex": rf"^.+-.+-({FLAGS.run_1}|{FLAGS.run_2})$"}},
    )

    assert len(runs) == 2, "No such runs"

    run_1, run_2 = runs
    config_1, config_2 = to_config_str(run_1), to_config_str(run_2)

    result = list(difflib.unified_diff(config_1, config_2))
    for line in result:
        line = line.rstrip("\n")
        if line.startswith("-"):
            print(Fore.RED + line + Style.RESET_ALL)
        elif line.startswith("+"):
            print(Fore.GREEN + line + Style.RESET_ALL)
        else:
            print(line)


if __name__ == "__main__":
    app.run(main)
