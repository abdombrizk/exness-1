#!/usr/bin/env python3
"""
Professional Logging System
===========================

Centralized logging system for the Gold Trading AI application.
Provides structured logging with multiple handlers, formatters, and log levels.

Author: AI Trading Systems
Version: 2.0.0
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors"""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
                
        return json.dumps(log_entry)


class TradingLogFilter(logging.Filter):
    """Custom filter for trading-specific logs"""
    
    def filter(self, record):
        """Filter trading-related log records"""
        trading_keywords = ['trade', 'signal', 'prediction', 'model', 'accuracy']
        message = record.getMessage().lower()
        
        # Add trading flag if message contains trading keywords
        record.is_trading = any(keyword in message for keyword in trading_keywords)
        
        return True


class LoggerManager:
    """Centralized logger management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logger manager
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        return {
            'level': 'INFO',
            'files': {
                'application': 'logs/application.log',
                'trading': 'logs/trading.log',
                'errors': 'logs/errors.log',
                'performance': 'logs/performance.log'
            },
            'rotation': {
                'max_size': '10MB',
                'backup_count': 5
            },
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S'
        }
        
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config['level'].upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup console handler
        self._setup_console_handler()
        
        # Setup file handlers
        self._setup_file_handlers()
        
    def _setup_console_handler(self):
        """Setup console handler with colored output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(
            fmt=self.config['format'],
            datefmt=self.config['date_format']
        )
        console_handler.setFormatter(console_formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(console_handler)
        
    def _setup_file_handlers(self):
        """Setup file handlers for different log types"""
        for log_type, log_file in self.config['files'].items():
            # Create rotating file handler
            handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=self._parse_size(self.config['rotation']['max_size']),
                backupCount=self.config['rotation']['backup_count'],
                encoding='utf-8'
            )
            
            # Set level based on log type
            if log_type == 'errors':
                handler.setLevel(logging.ERROR)
            elif log_type == 'performance':
                handler.setLevel(logging.DEBUG)
            else:
                handler.setLevel(logging.INFO)
                
            # Set formatter
            if log_type == 'trading':
                # Use JSON formatter for trading logs
                formatter = JSONFormatter()
                # Add trading filter
                handler.addFilter(TradingLogFilter())
            else:
                formatter = logging.Formatter(
                    fmt=self.config['format'],
                    datefmt=self.config['date_format']
                )
                
            handler.setFormatter(formatter)
            
            # Add to root logger
            logging.getLogger().addHandler(handler)
            
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes"""
        size_str = size_str.upper()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
            
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create logger with specified name
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
            
        return self.loggers[name]
        
    def set_level(self, level: str):
        """
        Set logging level for all loggers
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper())
        
        # Set root logger level
        logging.getLogger().setLevel(log_level)
        
        # Set level for all handlers
        for handler in logging.getLogger().handlers:
            if not isinstance(handler, logging.handlers.RotatingFileHandler) or \
               'errors' not in handler.baseFilename:
                handler.setLevel(log_level)
                
    def add_performance_log(self, operation: str, duration: float, **kwargs):
        """
        Add performance log entry
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional performance metrics
        """
        perf_logger = self.get_logger('performance')
        
        perf_data = {
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        perf_logger.info(f"Performance: {operation}", extra=perf_data)
        
    def add_trading_log(self, signal: str, confidence: float, price: float, **kwargs):
        """
        Add trading log entry
        
        Args:
            signal: Trading signal (BUY, SELL, HOLD)
            confidence: Model confidence
            price: Current price
            **kwargs: Additional trading data
        """
        trading_logger = self.get_logger('trading')
        
        trading_data = {
            'signal': signal,
            'confidence': confidence,
            'price': price,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        trading_logger.info(f"Trading Signal: {signal}", extra=trading_data)


# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None


def setup_logging(config: Optional[Dict[str, Any]] = None):
    """
    Setup logging system
    
    Args:
        config: Logging configuration dictionary
    """
    global _logger_manager
    _logger_manager = LoggerManager(config)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name: Logger name (default: calling module name)
        
    Returns:
        Logger instance
    """
    global _logger_manager
    
    # Initialize logger manager if not done
    if _logger_manager is None:
        setup_logging()
        
    # Use calling module name if no name provided
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
        
    return _logger_manager.get_logger(name)


def set_log_level(level: str):
    """
    Set logging level
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _logger_manager
    
    if _logger_manager is None:
        setup_logging()
        
    _logger_manager.set_level(level)


def log_performance(operation: str, duration: float, **kwargs):
    """
    Log performance metrics
    
    Args:
        operation: Operation name
        duration: Duration in seconds
        **kwargs: Additional metrics
    """
    global _logger_manager
    
    if _logger_manager is None:
        setup_logging()
        
    _logger_manager.add_performance_log(operation, duration, **kwargs)


def log_trading(signal: str, confidence: float, price: float, **kwargs):
    """
    Log trading activity
    
    Args:
        signal: Trading signal
        confidence: Model confidence
        price: Current price
        **kwargs: Additional trading data
    """
    global _logger_manager
    
    if _logger_manager is None:
        setup_logging()
        
    _logger_manager.add_trading_log(signal, confidence, price, **kwargs)


# Convenience decorators
def log_execution_time(logger_name: str = None):
    """Decorator to log function execution time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger = get_logger(logger_name or func.__module__)
                logger.debug(f"{func.__name__} executed in {duration:.4f} seconds")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                logger = get_logger(logger_name or func.__module__)
                logger.error(f"{func.__name__} failed after {duration:.4f} seconds: {e}")
                
                raise
                
        return wrapper
    return decorator


def log_exceptions(logger_name: str = None):
    """Decorator to log exceptions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(logger_name or func.__module__)
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise
                
        return wrapper
    return decorator
