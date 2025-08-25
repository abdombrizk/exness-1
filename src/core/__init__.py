"""
Core Components Module
=====================

This module contains the core business logic and components of the Gold Trading AI system:

- models: Machine learning models and training pipelines
- data: Data management and processing
- database: Database operations and management
- analysis: Technical and fundamental analysis engines
- risk: Risk management and position sizing

All core functionality is implemented here following enterprise patterns.
"""

from .models import ModelManager, AdvancedMLTrainer
from .data import DataManager, DataProcessor
from .database import DatabaseManager, TradingDatabase
from .analysis import TechnicalAnalyzer, FundamentalAnalyzer
from .risk import RiskManager

__all__ = [
    "ModelManager",
    "AdvancedMLTrainer", 
    "DataManager",
    "DataProcessor",
    "DatabaseManager",
    "TradingDatabase",
    "TechnicalAnalyzer",
    "FundamentalAnalyzer",
    "RiskManager"
]
