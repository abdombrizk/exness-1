#!/usr/bin/env python3
"""
Database Manager for Gold Trading AI
Handles persistence of predictions, performance metrics, and historical data

Author: AI Trading Systems
Version: 1.0.0
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Optional, Any


class DatabaseManager:
    """
    Database manager for Gold Trading AI system
    Handles SQLite database operations for predictions and performance tracking
    """
    
    def __init__(self, db_path: str = "data/trading_ai.db"):
        """
        Initialize database manager
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        
        # Ensure data directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:  # Only create directory if path has a directory component
            os.makedirs(db_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print(f"💾 Database Manager initialized: {db_path}")
    
    def connect(self) -> bool:
        """
        Connect to the database
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            print("✅ Database connection established")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.logger.error(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("📴 Database connection closed")
    
    def create_tables(self) -> bool:
        """
        Create necessary database tables
        
        Returns:
            bool: True if tables created successfully, False otherwise
        """
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    accuracy_estimate REAL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    position_size REAL,
                    risk_reward_ratio REAL,
                    win_probability REAL,
                    market_regime TEXT,
                    volatility_level TEXT,
                    technical_score INTEGER,
                    fundamental_score INTEGER,
                    risk_score INTEGER,
                    analysis_method TEXT,
                    model_predictions TEXT,  -- JSON string
                    confidence_factors TEXT  -- JSON string
                )
            ''')
            
            # Performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    prediction_id INTEGER,
                    actual_outcome TEXT,
                    profit_loss REAL,
                    accuracy_achieved REAL,
                    execution_time REAL,
                    data_quality_score REAL,
                    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
                )
            ''')
            
            # Market data cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    data_json TEXT NOT NULL,  -- JSON string of OHLCV data
                    expiry_time DATETIME NOT NULL
                )
            ''')
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_data TEXT  -- JSON string for additional data
                )
            ''')
            
            self.connection.commit()
            print("✅ Database tables created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Table creation failed: {e}")
            self.logger.error(f"Table creation error: {e}")
            return False
    
    def save_prediction(self, prediction: Dict[str, Any]) -> Optional[int]:
        """
        Save a prediction to the database
        
        Args:
            prediction (dict): Prediction data dictionary
            
        Returns:
            int: Prediction ID if successful, None otherwise
        """
        try:
            if not self.connection:
                if not self.connect():
                    return None
            
            cursor = self.connection.cursor()
            
            # Extract and prepare data
            model_predictions_json = json.dumps(prediction.get('model_predictions', {}))
            confidence_factors_json = json.dumps(prediction.get('confidence_factors', []))
            
            cursor.execute('''
                INSERT INTO predictions (
                    signal, confidence, accuracy_estimate, entry_price, stop_loss,
                    take_profit, position_size, risk_reward_ratio, win_probability,
                    market_regime, volatility_level, technical_score, fundamental_score,
                    risk_score, analysis_method, model_predictions, confidence_factors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.get('signal', 'UNKNOWN'),
                prediction.get('confidence', 0.0),
                prediction.get('accuracy_estimate', 0.0),
                prediction.get('entry_price', 0.0),
                prediction.get('stop_loss', 0.0),
                prediction.get('take_profit', 0.0),
                prediction.get('position_size', 0.0),
                prediction.get('risk_reward_ratio', 0.0),
                prediction.get('win_probability', 0.0),
                prediction.get('market_regime', 'UNKNOWN'),
                prediction.get('volatility_level', 'MODERATE'),
                prediction.get('technical_score', 50),
                prediction.get('fundamental_score', 50),
                prediction.get('risk_score', 50),
                prediction.get('analysis_method', 'Unknown'),
                model_predictions_json,
                confidence_factors_json
            ))
            
            prediction_id = cursor.lastrowid
            self.connection.commit()
            
            print(f"✅ Prediction saved with ID: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            print(f"❌ Failed to save prediction: {e}")
            self.logger.error(f"Prediction save error: {e}")
            return None
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent predictions from database
        
        Args:
            limit (int): Maximum number of predictions to retrieve
            
        Returns:
            list: List of prediction dictionaries
        """
        try:
            if not self.connection:
                if not self.connect():
                    return []
            
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            predictions = []
            
            for row in rows:
                prediction = dict(row)
                # Parse JSON fields
                if prediction['model_predictions']:
                    prediction['model_predictions'] = json.loads(prediction['model_predictions'])
                if prediction['confidence_factors']:
                    prediction['confidence_factors'] = json.loads(prediction['confidence_factors'])
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Failed to retrieve predictions: {e}")
            self.logger.error(f"Prediction retrieval error: {e}")
            return []
    
    def save_performance_metric(self, prediction_id: int, outcome: str, 
                              profit_loss: float, accuracy: float, 
                              execution_time: float = 0.0) -> bool:
        """
        Save performance metrics for a prediction
        
        Args:
            prediction_id (int): ID of the prediction
            outcome (str): Actual outcome (WIN/LOSS/PENDING)
            profit_loss (float): Profit or loss amount
            accuracy (float): Achieved accuracy
            execution_time (float): Execution time in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO performance (
                    prediction_id, actual_outcome, profit_loss, 
                    accuracy_achieved, execution_time
                ) VALUES (?, ?, ?, ?, ?)
            ''', (prediction_id, outcome, profit_loss, accuracy, execution_time))
            
            self.connection.commit()
            print(f"✅ Performance metric saved for prediction {prediction_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save performance metric: {e}")
            self.logger.error(f"Performance metric save error: {e}")
            return False
    
    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get performance statistics for the last N days
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            dict: Performance statistics
        """
        try:
            if not self.connection:
                if not self.connect():
                    return {}
            
            cursor = self.connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get basic stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(accuracy_estimate) as avg_accuracy_estimate,
                    COUNT(CASE WHEN signal = 'BUY' OR signal = 'STRONG_BUY' THEN 1 END) as buy_signals,
                    COUNT(CASE WHEN signal = 'SELL' OR signal = 'STRONG_SELL' THEN 1 END) as sell_signals,
                    COUNT(CASE WHEN signal = 'HOLD' THEN 1 END) as hold_signals
                FROM predictions 
                WHERE timestamp >= ?
            ''', (cutoff_date,))
            
            stats = dict(cursor.fetchone())
            
            # Get performance metrics if available
            cursor.execute('''
                SELECT 
                    COUNT(*) as completed_trades,
                    COUNT(CASE WHEN actual_outcome = 'WIN' THEN 1 END) as wins,
                    COUNT(CASE WHEN actual_outcome = 'LOSS' THEN 1 END) as losses,
                    AVG(profit_loss) as avg_profit_loss,
                    SUM(profit_loss) as total_profit_loss,
                    AVG(accuracy_achieved) as avg_accuracy_achieved
                FROM performance p
                JOIN predictions pr ON p.prediction_id = pr.id
                WHERE pr.timestamp >= ?
            ''', (cutoff_date,))
            
            perf_stats = dict(cursor.fetchone())
            stats.update(perf_stats)
            
            # Calculate win rate
            if stats['completed_trades'] > 0:
                stats['win_rate'] = stats['wins'] / stats['completed_trades'] * 100
            else:
                stats['win_rate'] = 0.0
            
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get performance stats: {e}")
            self.logger.error(f"Performance stats error: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 90) -> bool:
        """
        Clean up old data from database
        
        Args:
            days (int): Keep data newer than this many days
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Clean old market data cache
            cursor.execute('DELETE FROM market_data_cache WHERE timestamp < ?', (cutoff_date,))
            
            # Clean old system metrics
            cursor.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_date,))
            
            self.connection.commit()
            print(f"✅ Cleaned up data older than {days} days")
            return True
            
        except Exception as e:
            print(f"❌ Failed to cleanup old data: {e}")
            self.logger.error(f"Data cleanup error: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Example usage and testing
if __name__ == "__main__":
    # Test the database manager
    db = DatabaseManager("test_trading_ai.db")
    
    if db.connect():
        db.create_tables()
        
        # Test prediction save
        test_prediction = {
            'signal': 'BUY',
            'confidence': 85.0,
            'accuracy_estimate': 90.0,
            'entry_price': 2050.0,
            'stop_loss': 2030.0,
            'take_profit': 2080.0,
            'position_size': 0.5,
            'risk_reward_ratio': 2.0,
            'win_probability': 75.0,
            'market_regime': 'BULLISH',
            'volatility_level': 'MODERATE',
            'technical_score': 85,
            'fundamental_score': 80,
            'risk_score': 25,
            'analysis_method': 'Ensemble AI',
            'model_predictions': {'lstm': 0.8, 'cnn': 0.9},
            'confidence_factors': ['Strong momentum', 'Low volatility']
        }
        
        pred_id = db.save_prediction(test_prediction)
        if pred_id:
            print(f"Test prediction saved with ID: {pred_id}")
        
        # Test performance stats
        stats = db.get_performance_stats()
        print(f"Performance stats: {stats}")
        
        db.disconnect()
        print("Database test completed successfully!")
    else:
        print("Database test failed!")
