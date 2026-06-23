"""
Nocturne Core Module

Provides the main components for gesture-based music generation:
- VisionTracker: Hand and face detection
- MusicProcessor: Gesture to MIDI conversion
- MIDIDriver: MIDI output handling
- ConfigManager: Configuration management
- Logging: Application logging configuration
"""

from .vision import VisionTracker
from .processor import MusicProcessor
from .midi_driver import MIDIDriver
from .config import ConfigManager, get_config
from .logging_config import setup_logging, get_logger

__all__ = [
    "VisionTracker",
    "MusicProcessor", 
    "MIDIDriver",
    "ConfigManager",
    "get_config",
    "setup_logging",
    "get_logger"
]