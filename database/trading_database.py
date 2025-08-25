#!/usr/bin/env python3
"""
Robust Database System for Gold Trading AI
Comprehensive data storage and retrieval for predictions, performance, and historical data

Author: AI Trading Systems
Version: 2.0.0
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from contextlib import contextmanager


class TradingDatabase:
    """Comprehensive database system for trading data and predictions"""
    
    def __init__(self, db_path: str = "database/trading_data.db"):
        self.db_path = db_path
        self.ensure_database_directory()
        self.initialize_database()
        
    def ensure_database_directory(self):
        """Ensure database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
            
    def initialize_database(self):
        """Initialize database with all required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol VARCHAR(10) NOT NULL DEFAULT 'GOLD',
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    source VARCHAR(50) DEFAULT 'yfinance',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol)
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    model_version VARCHAR(20) DEFAULT '1.0',
                    signal VARCHAR(20) NOT NULL,
                    confidence REAL NOT NULL,
                    probability REAL NOT NULL,
                    current_price REAL NOT NULL,
                    features_json TEXT,
                    prediction_horizon INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name VARCHAR(100) NOT NULL,
                    model_version VARCHAR(20) DEFAULT '1.0',
                    evaluation_date DATE NOT NULL,
                    accuracy REAL NOT NULL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    training_samples INTEGER,
                    test_samples INTEGER,
                    feature_count INTEGER,
                    hyperparameters_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    position_size REAL NOT NULL,
                    risk_amount REAL,
                    expected_return REAL,
                    confidence REAL NOT NULL,
                    model_source VARCHAR(100),
                    status VARCHAR(20) DEFAULT 'OPEN',
                    exit_price REAL,
                    exit_timestamp DATETIME,
                    actual_return REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Risk management table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_management (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    portfolio_value REAL NOT NULL,
                    total_exposure REAL NOT NULL,
                    available_margin REAL NOT NULL,
                    risk_percentage REAL NOT NULL,
                    max_drawdown REAL,
                    var_95 REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Backtesting results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtesting_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name VARCHAR(100) NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_capital REAL NOT NULL,
                    total_return REAL NOT NULL,
                    annual_return REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    avg_win REAL NOT NULL,
                    avg_loss REAL NOT NULL,
                    largest_win REAL NOT NULL,
                    largest_loss REAL NOT NULL,
                    parameters_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON trading_signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_status ON trading_signals(status)')
            
            conn.commit()
            print("✅ Database initialized successfully")
            
    def store_market_data(self, data: Dict) -> bool:
        """Store market data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (timestamp, symbol, open_price, high_price, low_price, close_price, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['timestamp'],
                    data.get('symbol', 'GOLD'),
                    data['open'],
                    data['high'],
                    data['low'],
                    data['close'],
                    data['volume'],
                    data.get('source', 'yfinance')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"❌ Error storing market data: {e}")
            return False
            
    def store_prediction(self, prediction_data: Dict) -> bool:
        """Store ML prediction"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                features_json = json.dumps(prediction_data.get('features', {}))
                
                cursor.execute('''
                    INSERT INTO predictions 
                    (timestamp, model_name, model_version, signal, confidence, probability, 
                     current_price, features_json, prediction_horizon)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_data['timestamp'],
                    prediction_data['model_name'],
                    prediction_data.get('model_version', '1.0'),
                    prediction_data['signal'],
                    prediction_data['confidence'],
                    prediction_data['probability'],
                    prediction_data['current_price'],
                    features_json,
                    prediction_data.get('prediction_horizon', 1)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"❌ Error storing prediction: {e}")
            return False
            
    def store_model_performance(self, performance_data: Dict) -> bool:
        """Store model performance metrics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                hyperparams_json = json.dumps(performance_data.get('hyperparameters', {}))
                
                cursor.execute('''
                    INSERT INTO model_performance 
                    (model_name, model_version, evaluation_date, accuracy, precision_score, 
                     recall_score, f1_score, total_predictions, correct_predictions,
                     training_samples, test_samples, feature_count, hyperparameters_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance_data['model_name'],
                    performance_data.get('model_version', '1.0'),
                    performance_data['evaluation_date'],
                    performance_data['accuracy'],
                    performance_data.get('precision', None),
                    performance_data.get('recall', None),
                    performance_data.get('f1_score', None),
                    performance_data['total_predictions'],
                    performance_data['correct_predictions'],
                    performance_data.get('training_samples', None),
                    performance_data.get('test_samples', None),
                    performance_data.get('feature_count', None),
                    hyperparams_json
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"❌ Error storing model performance: {e}")
            return False
            
    def store_trading_signal(self, signal_data: Dict) -> int:
        """Store trading signal and return signal ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trading_signals 
                    (timestamp, signal_type, entry_price, stop_loss, take_profit, 
                     position_size, risk_amount, expected_return, confidence, model_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['timestamp'],
                    signal_data['signal_type'],
                    signal_data['entry_price'],
                    signal_data.get('stop_loss'),
                    signal_data.get('take_profit'),
                    signal_data['position_size'],
                    signal_data.get('risk_amount'),
                    signal_data.get('expected_return'),
                    signal_data['confidence'],
                    signal_data.get('model_source')
                ))
                
                signal_id = cursor.lastrowid
                conn.commit()
                return signal_id
                
        except Exception as e:
            print(f"❌ Error storing trading signal: {e}")
            return -1
            
    def update_trading_signal(self, signal_id: int, update_data: Dict) -> bool:
        """Update trading signal (e.g., when position is closed)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                set_clauses = []
                values = []
                
                for key, value in update_data.items():
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
                    
                values.append(signal_id)
                
                query = f"UPDATE trading_signals SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"❌ Error updating trading signal: {e}")
            return False

    def get_market_data(self, start_date: datetime = None, end_date: datetime = None,
                       symbol: str = 'GOLD') -> pd.DataFrame:
        """Retrieve market data"""
        try:
            with self.get_connection() as conn:
                query = '''
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume
                    FROM market_data
                    WHERE symbol = ?
                '''
                params = [symbol]

                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)

                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)

                query += ' ORDER BY timestamp'

                df = pd.read_sql_query(query, conn, params=params)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df

        except Exception as e:
            print(f"❌ Error retrieving market data: {e}")
            return pd.DataFrame()

    def get_predictions(self, start_date: datetime = None, end_date: datetime = None,
                       model_name: str = None) -> pd.DataFrame:
        """Retrieve predictions"""
        try:
            with self.get_connection() as conn:
                query = '''
                    SELECT timestamp, model_name, signal, confidence, probability,
                           current_price, prediction_horizon
                    FROM predictions
                    WHERE 1=1
                '''
                params = []

                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)

                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)

                if model_name:
                    query += ' AND model_name = ?'
                    params.append(model_name)

                query += ' ORDER BY timestamp DESC'

                df = pd.read_sql_query(query, conn, params=params)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df

        except Exception as e:
            print(f"❌ Error retrieving predictions: {e}")
            return pd.DataFrame()

    def get_model_performance(self, model_name: str = None) -> pd.DataFrame:
        """Retrieve model performance metrics"""
        try:
            with self.get_connection() as conn:
                query = '''
                    SELECT model_name, model_version, evaluation_date, accuracy,
                           precision_score, recall_score, f1_score, total_predictions,
                           correct_predictions, training_samples, test_samples, feature_count
                    FROM model_performance
                '''
                params = []

                if model_name:
                    query += ' WHERE model_name = ?'
                    params.append(model_name)

                query += ' ORDER BY evaluation_date DESC'

                df = pd.read_sql_query(query, conn, params=params)
                df['evaluation_date'] = pd.to_datetime(df['evaluation_date'])
                return df

        except Exception as e:
            print(f"❌ Error retrieving model performance: {e}")
            return pd.DataFrame()

    def get_trading_signals(self, start_date: datetime = None, end_date: datetime = None,
                           status: str = None) -> pd.DataFrame:
        """Retrieve trading signals"""
        try:
            with self.get_connection() as conn:
                query = '''
                    SELECT timestamp, signal_type, entry_price, stop_loss, take_profit,
                           position_size, risk_amount, expected_return, confidence,
                           status, exit_price, exit_timestamp, actual_return
                    FROM trading_signals
                    WHERE 1=1
                '''
                params = []

                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)

                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)

                if status:
                    query += ' AND status = ?'
                    params.append(status)

                query += ' ORDER BY timestamp DESC'

                df = pd.read_sql_query(query, conn, params=params)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if 'exit_timestamp' in df.columns:
                    df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])
                return df

        except Exception as e:
            print(f"❌ Error retrieving trading signals: {e}")
            return pd.DataFrame()

    def calculate_prediction_accuracy(self, model_name: str = None,
                                    days_back: int = 30) -> Dict:
        """Calculate prediction accuracy over time"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Get predictions
            predictions_df = self.get_predictions(start_date, end_date, model_name)

            if predictions_df.empty:
                return {'accuracy': 0.0, 'total_predictions': 0, 'correct_predictions': 0}

            # Get market data for the same period
            market_df = self.get_market_data(start_date, end_date)

            if market_df.empty:
                return {'accuracy': 0.0, 'total_predictions': 0, 'correct_predictions': 0}

            # Merge predictions with actual price movements
            merged_df = pd.merge_asof(
                predictions_df.sort_values('timestamp'),
                market_df.sort_values('timestamp'),
                left_on='timestamp',
                right_on='timestamp',
                direction='forward'
            )

            # Calculate actual outcomes
            correct_predictions = 0
            total_predictions = len(merged_df)

            for _, row in merged_df.iterrows():
                if pd.notna(row['close_price']) and pd.notna(row['current_price']):
                    actual_direction = 1 if row['close_price'] > row['current_price'] else 0
                    predicted_direction = 1 if row['signal'] in ['BUY', 'STRONG_BUY'] else 0

                    if actual_direction == predicted_direction:
                        correct_predictions += 1

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            return {
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'period_days': days_back
            }

        except Exception as e:
            print(f"❌ Error calculating prediction accuracy: {e}")
            return {'accuracy': 0.0, 'total_predictions': 0, 'correct_predictions': 0}

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            summary = {}

            # Model performance
            model_perf = self.get_model_performance()
            if not model_perf.empty:
                latest_perf = model_perf.iloc[0]
                summary['latest_model_accuracy'] = latest_perf['accuracy']
                summary['latest_model_name'] = latest_perf['model_name']
                summary['total_models'] = len(model_perf['model_name'].unique())
            else:
                summary['latest_model_accuracy'] = 0.0
                summary['latest_model_name'] = 'None'
                summary['total_models'] = 0

            # Prediction statistics
            recent_predictions = self.get_predictions(
                start_date=datetime.now() - timedelta(days=7)
            )
            summary['predictions_last_7_days'] = len(recent_predictions)

            # Trading signals
            recent_signals = self.get_trading_signals(
                start_date=datetime.now() - timedelta(days=30)
            )
            summary['signals_last_30_days'] = len(recent_signals)

            closed_signals = recent_signals[recent_signals['status'] == 'CLOSED']
            if not closed_signals.empty:
                profitable_signals = closed_signals[closed_signals['actual_return'] > 0]
                summary['win_rate'] = len(profitable_signals) / len(closed_signals)
                summary['avg_return'] = closed_signals['actual_return'].mean()
            else:
                summary['win_rate'] = 0.0
                summary['avg_return'] = 0.0

            # Data coverage
            market_data = self.get_market_data(
                start_date=datetime.now() - timedelta(days=30)
            )
            summary['market_data_points'] = len(market_data)

            return summary

        except Exception as e:
            print(f"❌ Error generating performance summary: {e}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to maintain database performance"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Clean up old market data
                cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_date,))
                market_deleted = cursor.rowcount

                # Clean up old predictions
                cursor.execute('DELETE FROM predictions WHERE timestamp < ?', (cutoff_date,))
                predictions_deleted = cursor.rowcount

                # Clean up old closed trading signals
                cursor.execute('''
                    DELETE FROM trading_signals
                    WHERE timestamp < ? AND status = 'CLOSED'
                ''', (cutoff_date,))
                signals_deleted = cursor.rowcount

                conn.commit()

                print(f"✅ Cleanup complete:")
                print(f"   Market data: {market_deleted} records")
                print(f"   Predictions: {predictions_deleted} records")
                print(f"   Trading signals: {signals_deleted} records")

                return True

        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            return False

    def export_data(self, table_name: str, file_path: str,
                   start_date: datetime = None, end_date: datetime = None) -> bool:
        """Export data to CSV file"""
        try:
            with self.get_connection() as conn:
                query = f'SELECT * FROM {table_name}'
                params = []

                if start_date and 'timestamp' in self.get_table_columns(table_name):
                    query += ' WHERE timestamp >= ?'
                    params.append(start_date)

                    if end_date:
                        query += ' AND timestamp <= ?'
                        params.append(end_date)
                elif end_date and 'timestamp' in self.get_table_columns(table_name):
                    query += ' WHERE timestamp <= ?'
                    params.append(end_date)

                df = pd.read_sql_query(query, conn, params=params)
                df.to_csv(file_path, index=False)

                print(f"✅ Exported {len(df)} records to {file_path}")
                return True

        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return False

    def get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for a table"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
                return columns

        except Exception as e:
            print(f"❌ Error getting table columns: {e}")
            return []

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Table sizes
                tables = ['market_data', 'predictions', 'model_performance',
                         'trading_signals', 'risk_management', 'backtesting_results']

                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    stats[f'{table}_count'] = count

                # Database file size
                stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)

                # Date ranges
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM market_data')
                result = cursor.fetchone()
                if result[0]:
                    stats['market_data_start'] = result[0]
                    stats['market_data_end'] = result[1]

                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM predictions')
                result = cursor.fetchone()
                if result[0]:
                    stats['predictions_start'] = result[0]
                    stats['predictions_end'] = result[1]

            return stats

        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}


class DatabaseManager:
    """High-level database manager with additional utilities"""

    def __init__(self, db_path: str = "database/trading_data.db"):
        self.db = TradingDatabase(db_path)

    def initialize_with_sample_data(self):
        """Initialize database with sample data for testing"""
        print("🔄 Initializing database with sample data...")

        # Generate sample market data
        self._generate_sample_market_data()

        # Generate sample predictions
        self._generate_sample_predictions()

        # Generate sample model performance
        self._generate_sample_model_performance()

        # Generate sample trading signals
        self._generate_sample_trading_signals()

        print("✅ Sample data initialization complete")

    def _generate_sample_market_data(self):
        """Generate sample market data"""
        import random

        # Generate 30 days of hourly data
        start_date = datetime.now() - timedelta(days=30)
        current_date = start_date
        base_price = 2000.0

        while current_date <= datetime.now():
            # Simulate realistic price movements
            change_pct = random.gauss(0, 0.01)  # 1% volatility
            base_price *= (1 + change_pct)

            # Generate OHLC
            open_price = base_price * (1 + random.gauss(0, 0.002))
            close_price = base_price * (1 + random.gauss(0, 0.002))
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.003)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.003)))
            volume = int(random.lognormvariate(10, 1))

            market_data = {
                'timestamp': current_date,
                'symbol': 'GOLD',
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'source': 'synthetic'
            }

            self.db.store_market_data(market_data)
            current_date += timedelta(hours=1)

    def _generate_sample_predictions(self):
        """Generate sample predictions"""
        import random

        models = ['RandomForest', 'XGBoost', 'LightGBM', 'Ensemble']
        signals = ['BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL']

        # Generate predictions for last 7 days
        start_date = datetime.now() - timedelta(days=7)
        current_date = start_date

        while current_date <= datetime.now():
            for model in models:
                if random.random() < 0.3:  # 30% chance of prediction per hour per model
                    prediction_data = {
                        'timestamp': current_date,
                        'model_name': model,
                        'model_version': '2.0',
                        'signal': random.choice(signals),
                        'confidence': random.uniform(0.6, 0.95),
                        'probability': random.uniform(0.4, 0.9),
                        'current_price': 2000 + random.gauss(0, 50),
                        'features': {'feature_count': random.randint(50, 150)},
                        'prediction_horizon': 1
                    }

                    self.db.store_prediction(prediction_data)

            current_date += timedelta(hours=1)

    def _generate_sample_model_performance(self):
        """Generate sample model performance data"""
        import random

        models = ['RandomForest', 'XGBoost', 'LightGBM', 'Ensemble']

        for model in models:
            performance_data = {
                'model_name': model,
                'model_version': '2.0',
                'evaluation_date': datetime.now().date(),
                'accuracy': random.uniform(0.85, 0.97),
                'precision': random.uniform(0.80, 0.95),
                'recall': random.uniform(0.80, 0.95),
                'f1_score': random.uniform(0.80, 0.95),
                'total_predictions': random.randint(1000, 5000),
                'correct_predictions': random.randint(800, 4500),
                'training_samples': random.randint(10000, 50000),
                'test_samples': random.randint(2000, 10000),
                'feature_count': random.randint(50, 150),
                'hyperparameters': {'n_estimators': 200, 'max_depth': 10}
            }

            self.db.store_model_performance(performance_data)

    def _generate_sample_trading_signals(self):
        """Generate sample trading signals"""
        import random

        signal_types = ['BUY', 'SELL']
        statuses = ['OPEN', 'CLOSED']

        # Generate signals for last 30 days
        start_date = datetime.now() - timedelta(days=30)
        current_date = start_date

        while current_date <= datetime.now():
            if random.random() < 0.1:  # 10% chance of signal per hour
                entry_price = 2000 + random.gauss(0, 50)
                signal_type = random.choice(signal_types)

                signal_data = {
                    'timestamp': current_date,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'stop_loss': entry_price * (0.98 if signal_type == 'BUY' else 1.02),
                    'take_profit': entry_price * (1.04 if signal_type == 'BUY' else 0.96),
                    'position_size': random.uniform(0.1, 2.0),
                    'risk_amount': random.uniform(100, 1000),
                    'expected_return': random.uniform(200, 2000),
                    'confidence': random.uniform(0.6, 0.9),
                    'model_source': random.choice(['RandomForest', 'XGBoost', 'LightGBM'])
                }

                signal_id = self.db.store_trading_signal(signal_data)

                # Randomly close some signals
                if random.random() < 0.7 and signal_id > 0:  # 70% chance of being closed
                    exit_price = entry_price * (1 + random.gauss(0, 0.02))
                    actual_return = (exit_price - entry_price) * signal_data['position_size']

                    if signal_type == 'SELL':
                        actual_return = -actual_return

                    update_data = {
                        'status': 'CLOSED',
                        'exit_price': exit_price,
                        'exit_timestamp': current_date + timedelta(hours=random.randint(1, 24)),
                        'actual_return': actual_return
                    }

                    self.db.update_trading_signal(signal_id, update_data)

            current_date += timedelta(hours=1)

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        print("📊 Generating performance report...")

        report = {}

        # Database statistics
        report['database_stats'] = self.db.get_database_stats()

        # Performance summary
        report['performance_summary'] = self.db.get_performance_summary()

        # Model accuracy over time
        report['model_accuracy'] = {}
        models = ['RandomForest', 'XGBoost', 'LightGBM', 'Ensemble']

        for model in models:
            accuracy_data = self.db.calculate_prediction_accuracy(model, days_back=30)
            report['model_accuracy'][model] = accuracy_data

        # Trading performance
        signals_df = self.db.get_trading_signals(
            start_date=datetime.now() - timedelta(days=30)
        )

        if not signals_df.empty:
            closed_signals = signals_df[signals_df['status'] == 'CLOSED']

            if not closed_signals.empty:
                report['trading_performance'] = {
                    'total_signals': len(signals_df),
                    'closed_signals': len(closed_signals),
                    'win_rate': len(closed_signals[closed_signals['actual_return'] > 0]) / len(closed_signals),
                    'avg_return': closed_signals['actual_return'].mean(),
                    'total_return': closed_signals['actual_return'].sum(),
                    'best_trade': closed_signals['actual_return'].max(),
                    'worst_trade': closed_signals['actual_return'].min()
                }
            else:
                report['trading_performance'] = {'message': 'No closed signals found'}
        else:
            report['trading_performance'] = {'message': 'No trading signals found'}

        return report

    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            import shutil
            shutil.copy2(self.db.db_path, backup_path)
            print(f"✅ Database backed up to {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

    def validate_data_integrity(self) -> Dict:
        """Validate database data integrity"""
        print("🔍 Validating data integrity...")

        validation_results = {
            'errors': [],
            'warnings': [],
            'summary': {}
        }

        try:
            # Check for missing required data
            market_data = self.db.get_market_data()
            if market_data.empty:
                validation_results['errors'].append("No market data found")
            else:
                validation_results['summary']['market_data_records'] = len(market_data)

            # Check for data consistency
            predictions = self.db.get_predictions()
            if not predictions.empty:
                validation_results['summary']['prediction_records'] = len(predictions)

                # Check for predictions without corresponding market data
                for _, pred in predictions.head(100).iterrows():  # Sample check
                    market_at_time = market_data[
                        (market_data['timestamp'] >= pred['timestamp'] - timedelta(minutes=30)) &
                        (market_data['timestamp'] <= pred['timestamp'] + timedelta(minutes=30))
                    ]

                    if market_at_time.empty:
                        validation_results['warnings'].append(
                            f"Prediction at {pred['timestamp']} has no corresponding market data"
                        )

            # Check model performance data
            model_perf = self.db.get_model_performance()
            if not model_perf.empty:
                validation_results['summary']['model_performance_records'] = len(model_perf)

                # Check for unrealistic accuracy values
                high_accuracy = model_perf[model_perf['accuracy'] > 0.99]
                if not high_accuracy.empty:
                    validation_results['warnings'].append(
                        f"{len(high_accuracy)} models have >99% accuracy (may be overfitted)"
                    )

            validation_results['summary']['total_errors'] = len(validation_results['errors'])
            validation_results['summary']['total_warnings'] = len(validation_results['warnings'])

            print(f"✅ Validation complete: {validation_results['summary']['total_errors']} errors, "
                  f"{validation_results['summary']['total_warnings']} warnings")

        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {e}")

        return validation_results


def main():
    """Main function for database testing"""
    print("🗄️  Gold Trading AI Database System")
    print("=" * 50)

    # Initialize database manager
    db_manager = DatabaseManager()

    # Initialize with sample data
    db_manager.initialize_with_sample_data()

    # Generate performance report
    report = db_manager.generate_performance_report()

    print("\n📊 Performance Report Summary:")
    print(f"   Database size: {report['database_stats'].get('database_size_mb', 0):.2f} MB")
    print(f"   Market data points: {report['database_stats'].get('market_data_count', 0)}")
    print(f"   Predictions: {report['database_stats'].get('predictions_count', 0)}")
    print(f"   Trading signals: {report['database_stats'].get('trading_signals_count', 0)}")

    if 'trading_performance' in report and 'win_rate' in report['trading_performance']:
        perf = report['trading_performance']
        print(f"   Win rate: {perf['win_rate']:.1%}")
        print(f"   Total return: ${perf['total_return']:.2f}")

    # Validate data integrity
    validation = db_manager.validate_data_integrity()
    print(f"\n🔍 Data Validation: {validation['summary']['total_errors']} errors, "
          f"{validation['summary']['total_warnings']} warnings")

    print("\n✅ Database system ready for production use!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
