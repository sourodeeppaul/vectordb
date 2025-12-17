"""
Logging utilities for VectorDB.
"""

import logging
import sys
from typing import Optional


# Default format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level logger cache
_loggers: dict = {}


def setup_logger(
    name: str = "vectordb",
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        log_file: Optional file path for logging
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        format_string or DEFAULT_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Cache the logger
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = "vectordb") -> logging.Logger:
    """
    Get a logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance (creates default if not exists)
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, *args):
        self.logger.setLevel(self.old_level)