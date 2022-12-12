from contextlib import contextmanager

import matplotlib.pyplot as plt


@contextmanager
def new_figure(show: bool = False, *args, **kwargs):
    fig = plt.figure(*args, **kwargs)

    yield fig

    if show:
        plt.show()

    plt.close(fig)
