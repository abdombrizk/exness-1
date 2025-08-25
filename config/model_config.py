#!/usr/bin/env python3
"""
Model Configuration for Gold Trading AI
Central configuration for all AI models and system parameters

Author: AI Trading Systems
Version: 1.0.0
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    # Target Performance
    'target_accuracy': 0.90,
    'target_precision': 0.85,
    'target_recall': 0.85,
    
    # Data Parameters
    'sequence_length': 60,
    'feature_dim': 50,
    'lookback_periods': {
        '1m': 100,
        '5m': 200,
        '15m': 300,
        '1h': 500,
        '4h': 200,
        '1d': 100
    },
    
    # Training Parameters
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 20,
    
    # Model Architecture
    'lstm_hidden_dim': 128,
    'lstm_num_layers': 3,
    'transformer_num_heads': 8,
    'transformer_dropout': 0.2,
    'cnn_num_filters': 64,
    'cnn_kernel_sizes': [3, 5, 7],
    'meta_learner_hidden_dim': 64,
    
    # Ensemble Weights
    'ensemble_weights': {
        'lstm_transformer': 0.3,
        'cnn_attention': 0.3,
        'random_forest': 0.2,
        'gradient_boost': 0.2
    }
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_risk_per_trade': 0.02,  # 2%
    'max_portfolio_risk': 0.10,  # 10%
    'max_drawdown': 0.05,        # 5%
    'position_size_limits': {
        'min_position': 0.01,
        'max_position_pct': 0.20  # 20% of portfolio
    },
    'stop_loss_config': {
        'atr_multiplier': 2.0,
        'max_stop_percentage': 0.03,  # 3%
        'min_stop_percentage': 0.005  # 0.5%
    },
    'take_profit_config': {
        'default_risk_reward': 2.0,
        'max_risk_reward': 3.0,
        'min_risk_reward': 1.5
    }
}

# Data Source Configuration
DATA_CONFIG = {
    'primary_symbol': 'GC=F',
    'alternative_symbols': ['XAUUSD=X', 'GLD'],
    'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
    'data_sources': {
        'market_data': 'yfinance',
        'fundamental_data': 'multiple',
        'sentiment_data': 'simulated'
    },
    'cache_timeout': 300,  # 5 minutes
    'max_retries': 3
}

# Technical Analysis Configuration
TECHNICAL_CONFIG = {
    'indicators': {
        'moving_averages': [5, 10, 20, 50, 100, 200],
        'rsi_periods': [14, 21, 30],
        'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
        'bollinger_params': {'period': 20, 'std_dev': 2},
        'atr_period': 14,
        'stochastic_params': {'k_period': 14, 'd_period': 3}
    },
    'pattern_recognition': True,
    'support_resistance': True,
    'fibonacci_levels': True
}

# Fundamental Analysis Configuration
FUNDAMENTAL_CONFIG = {
    'weights': {
        'dxy': 0.25,
        'fed_rates': 0.20,
        'inflation': 0.15,
        'geopolitical': 0.15,
        'oil': 0.10,
        'silver': 0.10,
        'bonds': 0.05
    },
    'update_frequency': 3600,  # 1 hour
    'data_sources': {
        'dxy': 'yfinance',
        'fed_rates': 'fred',
        'inflation': 'fred',
        'oil': 'yfinance',
        'silver': 'yfinance'
    }
}

# Performance Monitoring Configuration
PERFORMANCE_CONFIG = {
    'accuracy_threshold': 0.90,
    'response_time_threshold': 3.0,  # seconds
    'memory_threshold': 1024,  # MB
    'monitoring_interval': 60,  # seconds
    'alert_thresholds': {
        'accuracy_warning': 0.85,
        'accuracy_critical': 0.80,
        'response_time_warning': 5.0,
        'response_time_critical': 10.0
    },
    'performance_history_limit': 1000
}

# Database Configuration
DATABASE_CONFIG = {
    'db_path': 'database/gold_trading_ai.db',
    'backup_interval': 86400,  # 24 hours
    'cleanup_interval': 2592000,  # 30 days
    'max_records_per_table': 100000,
    'connection_timeout': 30
}

# GUI Configuration
GUI_CONFIG = {
    'window_title': 'Gold Trading AI - Professional Analysis',
    'window_size': '1400x900',
    'theme': 'dark',
    'update_interval': 5000,  # 5 seconds
    'chart_timeframes': ['1h', '4h', '1d'],
    'default_timeframe': '1h'
}

# API Configuration
API_CONFIG = {
    'rate_limits': {
        'yfinance': 2000,  # requests per hour
        'alpha_vantage': 500,
        'fred': 1000
    },
    'timeout': 30,  # seconds
    'retry_delay': 1,  # seconds
    'max_retries': 3
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': 'logs/gold_trading_ai.log',
    'max_file_size': 10485760,  # 10MB
    'backup_count': 5
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'technical_features': True,
    'fundamental_features': True,
    'sentiment_features': True,
    'pattern_features': True,
    'volatility_features': True,
    'time_features': True,
    'multi_timeframe_features': True,
    'feature_selection': {
        'method': 'correlation',
        'threshold': 0.95,
        'max_features': 100
    }
}

# Validation Configuration
VALIDATION_CONFIG = {
    'cross_validation_folds': 5,
    'test_size': 0.2,
    'validation_frequency': 86400,  # 24 hours
    'min_validation_samples': 1000,
    'confidence_interval': 0.95
}

# Export all configurations
__all__ = [
    'MODEL_CONFIG',
    'RISK_CONFIG', 
    'DATA_CONFIG',
    'TECHNICAL_CONFIG',
    'FUNDAMENTAL_CONFIG',
    'PERFORMANCE_CONFIG',
    'DATABASE_CONFIG',
    'GUI_CONFIG',
    'API_CONFIG',
    'LOGGING_CONFIG',
    'FEATURE_CONFIG',
    'VALIDATION_CONFIG'
]
