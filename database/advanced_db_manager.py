#!/usr/bin/env python3
"""
Advanced Database Manager for Gold Trading AI
Comprehensive data storage and retrieval system

Author: AI Trading Systems
Version: 1.0.0
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AdvancedDBManager:
    """
    Advanced database manager for gold trading AI system
    Handles storage and retrieval of market data, predictions, and performance metrics
    """
    
    def __init__(self, db_path='database/gold_trading_ai.db'):
        self.db_path = db_path
        self.connection = None
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        print(f"💾 Database Manager initialized: {db_path}")
        
    def _initialize_database(self):
        """Initialize database with required tables"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Create tables
            self._create_market_data_table()
            self._create_predictions_table()
            self._create_performance_table()
            self._create_fundamental_data_table()
            self._create_model_metrics_table()
            self._create_risk_metrics_table()
            
            self.connection.commit()
            print("✅ Database tables initialized")
            
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            
    def _create_market_data_table(self):
        """Create market data table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            timeframe VARCHAR(5) NOT NULL,
            open_price REAL NOT NULL,
            high_price REAL NOT NULL,
            low_price REAL NOT NULL,
            close_price REAL NOT NULL,
            volume REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol, timeframe)
        )
        """
        self.connection.execute(create_table_sql)
        
        # Create indexes
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe ON market_data(symbol, timeframe)"
        )
        
    def _create_predictions_table(self):
        """Create predictions table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            signal VARCHAR(20) NOT NULL,
            confidence REAL NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            position_size REAL NOT NULL,
            risk_reward_ratio REAL NOT NULL,
            technical_score REAL,
            fundamental_score REAL,
            risk_score REAL,
            model_accuracy REAL,
            market_regime VARCHAR(20),
            volatility_level VARCHAR(10),
            prediction_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.connection.execute(create_table_sql)
        
        # Create indexes
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_signal ON predictions(signal)"
        )
        
    def _create_performance_table(self):
        """Create performance tracking table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            prediction_id INTEGER,
            actual_outcome VARCHAR(20),
            profit_loss REAL,
            profit_loss_pct REAL,
            trade_duration_hours REAL,
            was_successful BOOLEAN,
            exit_reason VARCHAR(50),
            performance_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
        """
        self.connection.execute(create_table_sql)
        
        # Create indexes
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance(timestamp)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_performance_prediction_id ON performance(prediction_id)"
        )
        
    def _create_fundamental_data_table(self):
        """Create fundamental data table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS fundamental_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            data_type VARCHAR(50) NOT NULL,
            data_source VARCHAR(50) NOT NULL,
            value REAL,
            change_pct REAL,
            trend VARCHAR(20),
            impact_score REAL,
            fundamental_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.connection.execute(create_table_sql)
        
        # Create indexes
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_fundamental_timestamp ON fundamental_data(timestamp)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_fundamental_type ON fundamental_data(data_type)"
        )
        
    def _create_model_metrics_table(self):
        """Create model metrics table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            model_name VARCHAR(50) NOT NULL,
            accuracy REAL NOT NULL,
            precision_score REAL,
            recall SCORE REAL,
            f1_score REAL,
            validation_samples INTEGER,
            training_time_seconds REAL,
            model_version VARCHAR(20),
            metrics_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.connection.execute(create_table_sql)
        
        # Create indexes
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON model_metrics(timestamp)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON model_metrics(model_name)"
        )
        
    def _create_risk_metrics_table(self):
        """Create risk metrics table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS risk_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            portfolio_value REAL NOT NULL,
            total_risk_exposure REAL NOT NULL,
            current_drawdown REAL NOT NULL,
            max_drawdown REAL NOT NULL,
            active_positions INTEGER,
            risk_utilization REAL,
            risk_alerts TEXT,
            risk_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.connection.execute(create_table_sql)
        
        # Create indexes
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_risk_metrics_timestamp ON risk_metrics(timestamp)"
        )
        
    def store_market_data(self, market_data, symbol='XAUUSD'):
        """
        Store market data in database
        
        Args:
            market_data (dict): Market data for different timeframes
            symbol (str): Trading symbol
        """
        try:
            records_inserted = 0
            
            for timeframe, data in market_data.items():
                if data is None or len(data) == 0:
                    continue
                    
                # Prepare data for insertion
                for _, row in data.iterrows():
                    try:
                        insert_sql = """
                        INSERT OR REPLACE INTO market_data 
                        (timestamp, symbol, timeframe, open_price, high_price, low_price, close_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """
                        
                        timestamp = row.get('datetime', datetime.now())
                        if isinstance(timestamp, str):
                            timestamp = pd.to_datetime(timestamp)
                            
                        values = (
                            timestamp,
                            symbol,
                            timeframe,
                            float(row.get('open', 0)),
                            float(row.get('high', 0)),
                            float(row.get('low', 0)),
                            float(row.get('close', 0)),
                            float(row.get('volume', 0))
                        )
                        
                        self.connection.execute(insert_sql, values)
                        records_inserted += 1
                        
                    except Exception as e:
                        print(f"⚠️  Error inserting market data row: {e}")
                        continue
                        
            self.connection.commit()
            print(f"💾 Stored {records_inserted} market data records")
            
        except Exception as e:
            print(f"❌ Error storing market data: {e}")
            
    def store_prediction(self, prediction_result):
        """
        Store prediction result in database
        
        Args:
            prediction_result (dict): Prediction result from analyzer
            
        Returns:
            int: Prediction ID
        """
        try:
            insert_sql = """
            INSERT INTO predictions 
            (timestamp, signal, confidence, entry_price, stop_loss, take_profit, 
             position_size, risk_reward_ratio, technical_score, fundamental_score, 
             risk_score, model_accuracy, market_regime, volatility_level, prediction_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                datetime.now(),
                prediction_result.get('signal', 'UNKNOWN'),
                float(prediction_result.get('confidence', 0)),
                float(prediction_result.get('entry_price', 0)),
                float(prediction_result.get('stop_loss', 0)),
                float(prediction_result.get('take_profit', 0)),
                float(prediction_result.get('position_size', 0)),
                float(prediction_result.get('risk_reward_ratio', 0)),
                float(prediction_result.get('technical_score', 0)),
                float(prediction_result.get('fundamental_score', 0)),
                float(prediction_result.get('risk_score', 0)),
                float(prediction_result.get('accuracy_estimate', 0)),
                prediction_result.get('market_regime', 'UNKNOWN'),
                prediction_result.get('volatility_level', 'UNKNOWN'),
                json.dumps(prediction_result)
            )
            
            cursor = self.connection.execute(insert_sql, values)
            prediction_id = cursor.lastrowid
            self.connection.commit()
            
            print(f"💾 Stored prediction with ID: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            print(f"❌ Error storing prediction: {e}")
            return None
            
    def store_fundamental_data(self, fundamental_data):
        """
        Store fundamental data in database
        
        Args:
            fundamental_data (dict): Fundamental data from various sources
        """
        try:
            records_inserted = 0
            timestamp = datetime.now()
            
            for data_type, data_info in fundamental_data.items():
                if not isinstance(data_info, dict):
                    continue
                    
                try:
                    insert_sql = """
                    INSERT INTO fundamental_data 
                    (timestamp, data_type, data_source, value, change_pct, trend, impact_score, fundamental_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    values = (
                        timestamp,
                        data_type,
                        'API',  # Default source
                        float(data_info.get('current', 0)),
                        float(data_info.get('change_pct', 0)),
                        data_info.get('trend', 'UNKNOWN'),
                        float(data_info.get('impact_score', 50)),
                        json.dumps(data_info)
                    )
                    
                    self.connection.execute(insert_sql, values)
                    records_inserted += 1
                    
                except Exception as e:
                    print(f"⚠️  Error inserting fundamental data for {data_type}: {e}")
                    continue
                    
            self.connection.commit()
            print(f"💾 Stored {records_inserted} fundamental data records")
            
        except Exception as e:
            print(f"❌ Error storing fundamental data: {e}")
            
    def store_model_metrics(self, model_name, metrics):
        """
        Store model performance metrics
        
        Args:
            model_name (str): Name of the model
            metrics (dict): Model performance metrics
        """
        try:
            insert_sql = """
            INSERT INTO model_metrics 
            (timestamp, model_name, accuracy, precision_score, recall_score, f1_score, 
             validation_samples, training_time_seconds, model_version, metrics_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                datetime.now(),
                model_name,
                float(metrics.get('accuracy', 0)),
                float(metrics.get('precision', 0)),
                float(metrics.get('recall', 0)),
                float(metrics.get('f1_score', 0)),
                int(metrics.get('validation_samples', 0)),
                float(metrics.get('training_time', 0)),
                metrics.get('model_version', '1.0'),
                json.dumps(metrics)
            )
            
            self.connection.execute(insert_sql, values)
            self.connection.commit()
            
            print(f"💾 Stored model metrics for {model_name}")
            
        except Exception as e:
            print(f"❌ Error storing model metrics: {e}")
            
    def store_risk_metrics(self, risk_metrics):
        """
        Store risk management metrics
        
        Args:
            risk_metrics (dict): Risk metrics from risk manager
        """
        try:
            insert_sql = """
            INSERT INTO risk_metrics 
            (timestamp, portfolio_value, total_risk_exposure, current_drawdown, 
             max_drawdown, active_positions, risk_utilization, risk_alerts, risk_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                datetime.now(),
                float(risk_metrics.get('portfolio_value', 0)),
                float(risk_metrics.get('total_risk_amount', 0)),
                float(risk_metrics.get('current_drawdown', 0)),
                float(risk_metrics.get('max_drawdown_limit', 5)),
                int(risk_metrics.get('active_positions', 0)),
                float(risk_metrics.get('portfolio_risk_percentage', 0)),
                json.dumps(risk_metrics.get('risk_alerts', [])),
                json.dumps(risk_metrics)
            )
            
            self.connection.execute(insert_sql, values)
            self.connection.commit()
            
            print("💾 Stored risk metrics")
            
        except Exception as e:
            print(f"❌ Error storing risk metrics: {e}")
            
    def get_recent_predictions(self, limit=50):
        """
        Get recent predictions from database
        
        Args:
            limit (int): Number of recent predictions to retrieve
            
        Returns:
            pd.DataFrame: Recent predictions
        """
        try:
            query = """
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, self.connection, params=[limit])
            print(f"📊 Retrieved {len(df)} recent predictions")
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving predictions: {e}")
            return pd.DataFrame()
            
    def get_market_data(self, symbol='XAUUSD', timeframe='1h', days=30):
        """
        Get market data from database
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Data timeframe
            days (int): Number of days to retrieve
            
        Returns:
            pd.DataFrame: Market data
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
            SELECT * FROM market_data 
            WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(
                query, self.connection, 
                params=[symbol, timeframe, cutoff_date]
            )
            
            print(f"📊 Retrieved {len(df)} market data records")
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving market data: {e}")
            return pd.DataFrame()
            
    def get_performance_summary(self, days=30):
        """
        Get performance summary for specified period
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            dict: Performance summary
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get prediction statistics
            pred_query = """
            SELECT 
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence,
                AVG(model_accuracy) as avg_accuracy,
                AVG(technical_score) as avg_technical_score,
                AVG(fundamental_score) as avg_fundamental_score
            FROM predictions 
            WHERE timestamp >= ?
            """
            
            pred_stats = pd.read_sql_query(pred_query, self.connection, params=[cutoff_date])
            
            # Get signal distribution
            signal_query = """
            SELECT signal, COUNT(*) as count 
            FROM predictions 
            WHERE timestamp >= ?
            GROUP BY signal
            """
            
            signal_dist = pd.read_sql_query(signal_query, self.connection, params=[cutoff_date])
            
            # Get model metrics
            model_query = """
            SELECT 
                AVG(accuracy) as avg_model_accuracy,
                MAX(accuracy) as max_accuracy,
                MIN(accuracy) as min_accuracy
            FROM model_metrics 
            WHERE timestamp >= ?
            """
            
            model_stats = pd.read_sql_query(model_query, self.connection, params=[cutoff_date])
            
            summary = {
                'period_days': days,
                'prediction_stats': pred_stats.to_dict('records')[0] if not pred_stats.empty else {},
                'signal_distribution': signal_dist.to_dict('records') if not signal_dist.empty else [],
                'model_performance': model_stats.to_dict('records')[0] if not model_stats.empty else {},
                'generated_at': datetime.now().isoformat()
            }
            
            print(f"📊 Generated performance summary for {days} days")
            return summary
            
        except Exception as e:
            print(f"❌ Error generating performance summary: {e}")
            return {}
            
    def get_accuracy_history(self, model_name=None, days=90):
        """
        Get accuracy history for model validation
        
        Args:
            model_name (str): Specific model name (optional)
            days (int): Number of days to retrieve
            
        Returns:
            pd.DataFrame: Accuracy history
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            if model_name:
                query = """
                SELECT timestamp, model_name, accuracy, precision_score, recall_score, f1_score
                FROM model_metrics 
                WHERE model_name = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                """
                params = [model_name, cutoff_date]
            else:
                query = """
                SELECT timestamp, model_name, accuracy, precision_score, recall_score, f1_score
                FROM model_metrics 
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
                """
                params = [cutoff_date]
                
            df = pd.read_sql_query(query, self.connection, params=params)
            print(f"📊 Retrieved accuracy history: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving accuracy history: {e}")
            return pd.DataFrame()
            
    def cleanup_old_data(self, days_to_keep=365):
        """
        Clean up old data to manage database size
        
        Args:
            days_to_keep (int): Number of days of data to keep
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Tables to clean up
            tables = ['market_data', 'predictions', 'performance', 'fundamental_data', 'model_metrics', 'risk_metrics']
            
            total_deleted = 0
            for table in tables:
                delete_query = f"DELETE FROM {table} WHERE timestamp < ?"
                cursor = self.connection.execute(delete_query, [cutoff_date])
                deleted_rows = cursor.rowcount
                total_deleted += deleted_rows
                print(f"   Deleted {deleted_rows} old records from {table}")
                
            self.connection.commit()
            
            # Vacuum database to reclaim space
            self.connection.execute("VACUUM")
            
            print(f"🧹 Cleanup complete: {total_deleted} total records deleted")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            
    def export_data(self, table_name, output_file, days=30):
        """
        Export data to CSV file
        
        Args:
            table_name (str): Name of table to export
            output_file (str): Output CSV file path
            days (int): Number of days to export
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = f"""
            SELECT * FROM {table_name} 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, self.connection, params=[cutoff_date])
            df.to_csv(output_file, index=False)
            
            print(f"📤 Exported {len(df)} records from {table_name} to {output_file}")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            
    def get_database_stats(self):
        """
        Get database statistics
        
        Returns:
            dict: Database statistics
        """
        try:
            stats = {}
            
            # Get table sizes
            tables = ['market_data', 'predictions', 'performance', 'fundamental_data', 'model_metrics', 'risk_metrics']
            
            for table in tables:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = self.connection.execute(count_query).fetchone()
                stats[f'{table}_count'] = result[0]
                
            # Get database file size
            if os.path.exists(self.db_path):
                file_size = os.path.getsize(self.db_path)
                stats['database_size_mb'] = round(file_size / (1024 * 1024), 2)
            else:
                stats['database_size_mb'] = 0
                
            # Get date range
            date_query = """
            SELECT 
                MIN(timestamp) as earliest_data,
                MAX(timestamp) as latest_data
            FROM predictions
            """
            
            date_result = self.connection.execute(date_query).fetchone()
            if date_result[0]:
                stats['earliest_data'] = date_result[0]
                stats['latest_data'] = date_result[1]
                
            stats['generated_at'] = datetime.now().isoformat()
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}
            
    def close_connection(self):
        """Close database connection"""
        try:
            if self.connection:
                self.connection.close()
                print("💾 Database connection closed")
        except Exception as e:
            print(f"❌ Error closing database: {e}")
            
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close_connection()
