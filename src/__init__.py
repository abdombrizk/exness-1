"""
Gold Trading AI - Professional Trading Analysis System
======================================================

A comprehensive machine learning system for gold trading prediction with
professional Bloomberg Terminal-style interface, achieving >90% accuracy
through advanced ensemble methods.

Author: AI Trading Systems
Version: 2.0.0
License: MIT
"""

__version__ = "2.0.0"
__author__ = "AI Trading Systems"
__email__ = "support@goldtradingai.com"
__license__ = "MIT"

# Package metadata
__title__ = "Gold Trading AI"
__description__ = "Professional ML system for gold trading prediction"
__url__ = "https://github.com/your-repo/gold-trading-ai"

# Version info
VERSION_INFO = {
    "major": 2,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

# API imports for easy access
from .core.models import ModelManager
from .core.data import DataManager
from .core.database import DatabaseManager
from .gui.application import GoldTradingApp
from .utils.config import ConfigManager
from .utils.logger import get_logger

__all__ = [
    "ModelManager",
    "DataManager", 
    "DatabaseManager",
    "GoldTradingApp",
    "ConfigManager",
    "get_logger"
]
