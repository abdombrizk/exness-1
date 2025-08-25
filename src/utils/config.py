#!/usr/bin/env python3
"""
Configuration Management System
==============================

Centralized configuration management for the Gold Trading AI system.
Handles loading, validation, and access to configuration settings.

Author: AI Trading Systems
Version: 2.0.0
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    type: str = "sqlite"
    path: str = "data/database/trading_data.db"
    backup_path: str = "data/backups/"
    cleanup_days: int = 365
    backup_frequency: str = "daily"
    validation_enabled: bool = True
    connection_pool_size: int = 10
    query_timeout: int = 30


@dataclass
class MLConfig:
    """Machine learning configuration settings"""
    target_accuracy: float = 0.90
    confidence_threshold: float = 0.60
    ensemble_weights: str = "auto"
    train_test_split: float = 0.8
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    feature_count: int = 129
    lookback_periods: list = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])


@dataclass
class GUIConfig:
    """GUI configuration settings"""
    theme: str = "professional_dark"
    font_family: str = "Segoe UI"
    window_size: list = field(default_factory=lambda: [1600, 1000])
    min_window_size: list = field(default_factory=lambda: [1200, 800])
    split_ratio: float = 0.5
    auto_refresh: bool = True
    refresh_interval: int = 30
    chart_timeframe: str = "1h"


@dataclass
class DataSourceConfig:
    """Data source configuration settings"""
    primary: str = "yfinance"
    fallback: str = "synthetic"
    symbols: list = field(default_factory=lambda: ["GC=F", "XAUUSD=X", "GLD"])
    real_time_interval: int = 30
    historical_interval: int = 3600
    alpha_vantage_key: Optional[str] = None
    quandl_key: Optional[str] = None


@dataclass
class RiskConfig:
    """Risk management configuration settings"""
    default_position_size: float = 1.0
    max_position_size: float = 10.0
    risk_per_trade: float = 0.02
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04
    trailing_stop_enabled: bool = True
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.20
    max_correlation: float = 0.70


@dataclass
class Settings:
    """Main settings container"""
    application: Dict[str, Any] = field(default_factory=dict)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    machine_learning: MLConfig = field(default_factory=MLConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    risk_management: RiskConfig = field(default_factory=RiskConfig)
    technical_analysis: Dict[str, Any] = field(default_factory=dict)
    fundamental_analysis: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    testing: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    deployment: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Configuration manager for the Gold Trading AI system"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (default: config/settings.yaml)
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = config_path or self.project_root / "config" / "settings.yaml"
        self.settings = Settings()
        self._load_configuration()
        
    def _load_configuration(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.suffix.lower() == '.yaml':
                        config_data = yaml.safe_load(f)
                    elif self.config_path.suffix.lower() == '.json':
                        config_data = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
                        
                self._parse_configuration(config_data)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                logger.info("Using default configuration")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            
    def _parse_configuration(self, config_data: Dict[str, Any]):
        """Parse configuration data into settings objects"""
        try:
            # Application settings
            if 'application' in config_data:
                self.settings.application = config_data['application']
                
            # Database settings
            if 'database' in config_data:
                db_config = config_data['database']
                self.settings.database = DatabaseConfig(**db_config)
                
            # Machine learning settings
            if 'machine_learning' in config_data:
                ml_config = config_data['machine_learning']
                # Extract basic settings
                basic_ml_config = {k: v for k, v in ml_config.items() 
                                 if k not in ['models']}
                self.settings.machine_learning = MLConfig(**basic_ml_config)
                
            # GUI settings
            if 'gui' in config_data:
                gui_config = config_data['gui']
                # Extract basic settings
                basic_gui_config = {k: v for k, v in gui_config.items() 
                                  if k not in ['colors', 'font_sizes']}
                self.settings.gui = GUIConfig(**basic_gui_config)
                
            # Data sources settings
            if 'data_sources' in config_data:
                ds_config = config_data['data_sources']
                self.settings.data_sources = DataSourceConfig(**ds_config)
                
            # Risk management settings
            if 'risk_management' in config_data:
                risk_config = config_data['risk_management']
                self.settings.risk_management = RiskConfig(**risk_config)
                
            # Store other settings as dictionaries
            for key in ['technical_analysis', 'fundamental_analysis', 'logging', 
                       'performance', 'testing', 'security', 'deployment']:
                if key in config_data:
                    setattr(self.settings, key, config_data[key])
                    
        except Exception as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'database.path')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            value = self.settings
            
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
                    
            return value
            
        except Exception:
            return default
            
    def set(self, key: str, value: Any):
        """
        Set configuration value by key
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        try:
            keys = key.split('.')
            target = self.settings
            
            # Navigate to parent
            for k in keys[:-1]:
                if hasattr(target, k):
                    target = getattr(target, k)
                elif isinstance(target, dict):
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                else:
                    raise ValueError(f"Cannot set key: {key}")
                    
            # Set final value
            final_key = keys[-1]
            if hasattr(target, final_key):
                setattr(target, final_key, value)
            elif isinstance(target, dict):
                target[final_key] = value
            else:
                raise ValueError(f"Cannot set key: {key}")
                
        except Exception as e:
            logger.error(f"Error setting configuration key {key}: {e}")
            raise
            
    def save(self, path: Optional[Union[str, Path]] = None):
        """
        Save configuration to file
        
        Args:
            path: Path to save configuration (default: current config path)
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            # Convert settings to dictionary
            config_data = self._settings_to_dict()
            
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on file extension
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.suffix.lower() == '.yaml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {save_path.suffix}")
                    
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
            
    def _settings_to_dict(self) -> Dict[str, Any]:
        """Convert settings object to dictionary"""
        config_data = {}
        
        # Convert dataclass fields to dictionaries
        for field_name in ['application', 'technical_analysis', 'fundamental_analysis',
                          'logging', 'performance', 'testing', 'security', 'deployment']:
            value = getattr(self.settings, field_name)
            if isinstance(value, dict):
                config_data[field_name] = value
                
        # Convert dataclass objects
        for field_name, field_obj in [
            ('database', self.settings.database),
            ('machine_learning', self.settings.machine_learning),
            ('gui', self.settings.gui),
            ('data_sources', self.settings.data_sources),
            ('risk_management', self.settings.risk_management)
        ]:
            if hasattr(field_obj, '__dict__'):
                config_data[field_name] = field_obj.__dict__
                
        return config_data
        
    def validate(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate required paths exist
            required_dirs = [
                self.get('database.path', '').rsplit('/', 1)[0],
                self.get('database.backup_path', ''),
                'logs',
                'data'
            ]
            
            for dir_path in required_dirs:
                if dir_path:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    
            # Validate numeric ranges
            if not 0.5 <= self.get('machine_learning.target_accuracy', 0.9) <= 1.0:
                logger.error("Invalid target_accuracy: must be between 0.5 and 1.0")
                return False
                
            if not 0.0 <= self.get('machine_learning.confidence_threshold', 0.6) <= 1.0:
                logger.error("Invalid confidence_threshold: must be between 0.0 and 1.0")
                return False
                
            # Validate GUI settings
            window_size = self.get('gui.window_size', [1600, 1000])
            if len(window_size) != 2 or window_size[0] < 800 or window_size[1] < 600:
                logger.error("Invalid window_size: must be [width, height] with minimum 800x600")
                return False
                
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
            
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for sensitive configuration"""
        env_vars = {}
        
        # API keys
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            env_vars['ALPHA_VANTAGE_API_KEY'] = os.getenv('ALPHA_VANTAGE_API_KEY')
            
        if os.getenv('QUANDL_API_KEY'):
            env_vars['QUANDL_API_KEY'] = os.getenv('QUANDL_API_KEY')
            
        # Database settings
        if os.getenv('DATABASE_URL'):
            env_vars['DATABASE_URL'] = os.getenv('DATABASE_URL')
            
        return env_vars
        
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(config_path={self.config_path})"
        
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"ConfigManager(config_path={self.config_path}, settings={self.settings})"


# Global configuration instance
config = ConfigManager()

# Convenience functions
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config.get(key, default)

def set_config(key: str, value: Any):
    """Set configuration value"""
    config.set(key, value)

def validate_config() -> bool:
    """Validate configuration"""
    return config.validate()
