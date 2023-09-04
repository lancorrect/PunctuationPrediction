import datetime
import logging
import os
import sys
import termcolor
from pathlib import Path

__all__ = ['setup_logger']

logger_initialized = []
output_initialized = []


def setup_logger(name, output=None):
    """
    Initialize logger and set its verbosity level to INFO.
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = ("[%(asctime2)s %(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s - %(message2)s")
    color_formatter = ColoredFormatter(formatter, datefmt="%m/%d %H:%M:%S")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(color_formatter)
    logger.addHandler(ch)

    # file logging: all workers
    if output_initialized or output is not None:
        filename = output_initialized[0] if output_initialized else output
        if filename.endswith(".txt") or filename.endswith(".log"):
                pass
        else:
            filename = os.path.join(filename, ".log")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter())
        logger.addHandler(fh)
        if not output_initialized:
            output_initialized.append(filename)
    logger_initialized.append(name)
    return logger


COLORS = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt, use_color=True):
        logging.Formatter.__init__(self, fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:

            def colored(text):
                return termcolor.colored(
                    text,
                    color=COLORS[levelname],
                    attrs={"bold": True},
                )

            record.levelname2 = colored("{:<7}".format(record.levelname))
            record.message2 = colored(record.msg)

            asctime2 = datetime.datetime.fromtimestamp(record.created)
            record.asctime2 = termcolor.colored(asctime2, color="green")

            record.module2 = termcolor.colored(record.module, color="cyan")
            record.funcName2 = termcolor.colored(record.funcName, color="cyan")
            record.lineno2 = termcolor.colored(record.lineno, color="cyan")
        return logging.Formatter.format(self, record)

