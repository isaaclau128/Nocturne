"""
Configuration module for Nocturne.

Loads and manages application settings from config.json.
"""

import json
import os
from pathlib import Path


class ConfigManager:
    """Manages application configuration from JSON file."""
    
    def __init__(self, config_path=None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to config.json. Defaults to project root.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded config from {self.config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def get(self, key, default=None):
        """
        Get a config value using dot notation.
        
        Examples:
            config.get('vision.hand_detection_confidence')
            config.get('music.solfeggio_notes')
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_section(self, section):
        """Get an entire config section as a dictionary."""
        return self.config.get(section, {})


# Global config instance
_config = None


def get_config(config_path=None):
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = ConfigManager(config_path)
    return _config
