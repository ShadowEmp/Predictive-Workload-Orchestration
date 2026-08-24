"""
config/logger.py
================
Centralised logging factory.

Usage (in any module):
    from config.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys
from pathlib import Path

from config.settings import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that writes to both stdout and a rotating log file.

    Parameters
    ----------
    name : str
        Typically passed as ``__name__`` from the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called more than once
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    return logger
