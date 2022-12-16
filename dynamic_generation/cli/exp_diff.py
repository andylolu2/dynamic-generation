import difflib

from absl import app, flags, logging
from colorama import Fore, Style
from ml_collections import ConfigDict

from dynamic_generation.utils.secrets import secrets
from dynamic_generation.utils.wandb import load_runs

FLAGS = flags.FLAGS

flags.DEFINE_string("project", None, "The project name.")
flags.DEFINE_string("run_1", None, "The run number of the first run.")
flags.DEFINE_string("run_2", None, "The run number of the second run.")

flags.mark_flags_as_required(["project", "run_1", "run_2"])


def to_config_str(run):
    config = ConfigDict(run.config)
    return str(config).splitlines(keepends=True)


def main(argv):
    logging.level_info()

    run_1, run_2 = load_runs(
        secrets.wandb.entity, FLAGS.project, [FLAGS.run_1, FLAGS.run_2]
    )
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
