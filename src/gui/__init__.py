"""
GUI Components Module
====================

This module contains all graphical user interface components:

- application: Main application window and controller
- components: Reusable UI components and widgets
- themes: Professional themes and styling
- charts: Chart components and visualizations
- dialogs: Dialog windows and forms

Implements Bloomberg Terminal-style professional interface.
"""

from .application import GoldTradingApp
from .components import ProfessionalTheme, DataDisplayWidget, ChartWidget
from .themes import ThemeManager
from .charts import PriceChart, IndicatorChart
from .dialogs import SettingsDialog, AboutDialog

__all__ = [
    "GoldTradingApp",
    "ProfessionalTheme",
    "DataDisplayWidget", 
    "ChartWidget",
    "ThemeManager",
    "PriceChart",
    "IndicatorChart",
    "SettingsDialog",
    "AboutDialog"
]
