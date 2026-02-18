"""
Logging configuration and utilities
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import settings


def setup_logger(name: str = None, log_file: Optional[Path] = None,
                 level: str = None) -> logging.Logger:
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name (defaults to root logger)
        log_file: Path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    # Get logger
    logger = logging.getLogger(name)

    # Set level
    log_level = getattr(logging, level or settings.LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file or settings.LOG_FILE:
        file_path = log_file or settings.LOG_FILE
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger by name"""
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary log level changes"""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)