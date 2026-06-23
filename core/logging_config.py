"""
Logging configuration for Nocturne.

Sets up console and file logging with configurable levels.
"""

import logging
import logging.handlers
from pathlib import Path
from .config import get_config


def setup_logging(name=None):
    """
    Setup logging with both console and file handlers.
    
    Args:
        name: Logger name (usually __name__). Defaults to 'nocturne'.
    
    Returns:
        logging.Logger instance
    """
    if name is None:
        name = 'nocturne'
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    try:
        config = get_config()
        log_config = config.get_section('logging')
        
        level_str = log_config.get('level', 'INFO')
        log_file = log_config.get('log_file', 'nocturne.log')
        max_bytes = log_config.get('max_bytes', 5242880)  # 5MB
        backup_count = log_config.get('backup_count', 3)
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        level = getattr(logging, level_str.upper(), logging.INFO)
    except Exception as e:
        print(f"Warning: Could not load logging config: {e}. Using defaults.")
        level = logging.INFO
        log_file = 'nocturne.log'
        max_bytes = 5242880
        backup_count = 3
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger.setLevel(level)
    
    # Format
    formatter = logging.Formatter(format_str)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler with rotation
    try:
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not setup file logging: {e}")
    
    return logger


# Global logger
_logger = None


def get_logger(name=None):
    """Get or create the global logger."""
    global _logger
    if _logger is None:
        _logger = setup_logging(name or 'nocturne')
    return _logger
