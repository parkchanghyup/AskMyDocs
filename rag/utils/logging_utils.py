import logging
import sys
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with the specified name, level, and optional log file
    
    Args:
        name: The name of the logger
        level: The logging level (default: INFO)
        log_file: Optional path to a log file
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 