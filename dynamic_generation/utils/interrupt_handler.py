import signal
from typing import Any, Callable

from absl import logging


class Breakout(Exception):
    pass


class InterruptHandler:
    """
    Catches a KeyboardInterrupt and runs a custom handler.

    Implementation taken from University of Cambridge Probabilistic ML notes.
    """

    def __init__(self, handler: Callable[[], Any]):
        self.interrupted = False
        self.original_handler = None
        self.handler = handler

    def __enter__(self):
        self.original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handle)
        return self.check

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.original_handler)
        if exc_type == Breakout:
            self.handler()
            return True
        return False

    def handle(self, signal, frame):
        if self.interrupted:
            self.original_handler(signal, frame)  # type: ignore
        logging.info("Caught interrupt...")
        self.interrupted = True

    def check(self):
        if self.interrupted:
            raise Breakout
