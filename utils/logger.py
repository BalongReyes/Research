"""
utils/logger.py
===============
Centralised logging configuration for the Cobb 500 Chick Defect System.

Call setup_logging() once at application startup (in main.py).
All other modules use:  logger = logging.getLogger(__name__)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging(level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """
    Configure root logger with console + optional rotating file handler.

    Parameters
    ----------
    level       : "DEBUG" | "INFO" | "WARNING" | "ERROR"
    log_to_file : If True, writes to logs/system_<timestamp>.log

    Returns
    -------
    Root logger instance
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    if log_to_file:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"system_{ts}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info(f"Logging to file: {log_file}")

    return root
