"""
Logging utilities for ChatCFD.

Provides consistent logging configuration across all modules.

Usage:
    from logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("Processing file...")
    logger.warning("Something unexpected")
    logger.error("Failed to process")
"""

import logging
import sys
from typing import Optional

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track if root logger has been configured
_root_configured = False


def configure_logging(
    level: int = logging.INFO,
    format_str: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    stream: Optional[object] = None,
) -> None:
    """
    Configure the root logger for ChatCFD.

    Call this once at application startup to set up logging.
    Subsequent calls will be ignored.

    Args:
        level: Logging level (default: INFO)
        format_str: Log message format
        date_format: Date format for timestamps
        stream: Output stream (default: sys.stderr)
    """
    global _root_configured
    if _root_configured:
        return

    root = logging.getLogger("chatcfd")
    root.setLevel(level)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_str, date_format))

    root.addHandler(handler)
    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance configured under the chatcfd hierarchy.
    """
    # Ensure root is configured with defaults
    configure_logging()

    # Strip src2. prefix if present for cleaner names
    if name.startswith("src2."):
        name = name[5:]

    return logging.getLogger(f"chatcfd.{name}")


def set_verbose(verbose: bool) -> None:
    """
    Set verbose mode for all ChatCFD loggers.

    Args:
        verbose: If True, set level to DEBUG. If False, set to INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("chatcfd").setLevel(level)


class LoggerAdapter:
    """
    Adapter that provides print-like interface but uses logging.

    Useful for gradual migration from print() to logging.

    Usage:
        log = LoggerAdapter(get_logger(__name__), verbose=True)
        log("This is logged at INFO level")
        log.debug("This is logged at DEBUG level")
    """

    def __init__(self, logger: logging.Logger, verbose: bool = True):
        self.logger = logger
        self.verbose = verbose

    def __call__(self, message: str) -> None:
        """Log at INFO level when called like a function."""
        if self.verbose:
            self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log at DEBUG level."""
        if self.verbose:
            self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log at INFO level."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log at WARNING level."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log at ERROR level."""
        self.logger.error(message)
