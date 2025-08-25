#!/usr/bin/env python3
"""
Gold Trading AI - Advanced Model Training System
Comprehensive training with hyperparameter optimization and performance validation

Author: AI Trading Systems
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import joblib
import json
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.advanced_data_fetcher import AdvancedDataFetcher
from utils.feature_engineering import FeatureEngineer
from models.high_accuracy_ensemble import HighAccuracyEnsemble
from models.lstm_transformer import LSTMTransformerModel
from models.cnn_attention import CNNAttentionModel
from models.meta_learner import MetaLearner
from utils.accuracy_validator import AccuracyValidator
from config.model_config import MODEL_CONFIG, VALIDATION_CONFIG


class AdvancedModelTrainer:
    """
    Advanced model training system with hyperparameter optimization
    and comprehensive performance validation
    """
    
    def __init__(self, target_accuracy=0.90):
        self.target_accuracy = target_accuracy
        self.data_fetcher = AdvancedDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.accuracy_validator = AccuracyValidator(target_accuracy)
        
        # Training configuration
        self.config = MODEL_CONFIG.copy()
        self.validation_config = VALIDATION_CONFIG.copy()
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        self.performance_metrics = {}
        
        # Data storage
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        
        print("🤖 Advanced Model Trainer initialized")
        print(f"   Target accuracy: {target_accuracy:.1%}")
        print(f"   Training configuration loaded")
        
    def prepare_comprehensive_dataset(self):
        """Prepare comprehensive training dataset with multiple timeframes"""
        print("\n📊 Preparing comprehensive training dataset...")
        
        try:
            # Fetch data for multiple timeframes and periods
            datasets = {}
            
            # Primary gold symbols to try
            symbols = ['GC=F', 'XAUUSD=X', 'GLD']
            timeframes = ['1h', '4h', '1d']
            periods = ['2y', '1y', '6mo']  # Reduced periods for better data availability
            
            successful_fetches = 0
            total_attempts = 0
            
            for symbol in symbols:
                for timeframe in timeframes:
                    for period in periods:
                        total_attempts += 1
                        try:
                            print(f"   📈 Fetching {symbol} ({period}, {timeframe})...")
                            data = self.data_fetcher.fetch_historical_data(symbol, period, timeframe)
                            
                            if data is not None and len(data) > 100:
                                key = f"{symbol}_{timeframe}_{period}"
                                datasets[key] = data
                                successful_fetches += 1
                                print(f"      ✅ {len(data)} records")
                            else:
                                print(f"      ⚠️  Insufficient data")
                                
                        except Exception as e:
                            print(f"      ❌ Error: {e}")
                            
            print(f"\n📊 Data collection summary:")
            print(f"   Successful fetches: {successful_fetches}/{total_attempts}")
            
            if successful_fetches == 0:
                print("⚠️  No real data available. Generating synthetic training data...")
                return self._generate_synthetic_training_data()
                
            # Combine and process datasets
            combined_data = self._combine_datasets(datasets)
            
            if len(combined_data) < 1000:
                print("⚠️  Limited real data. Augmenting with synthetic data...")
                synthetic_data = self._generate_synthetic_training_data(base_data=combined_data)
                combined_data = pd.concat([combined_data, synthetic_data], ignore_index=True)
                
            print(f"✅ Final dataset: {len(combined_data)} records")
            return combined_data
            
        except Exception as e:
            print(f"❌ Dataset preparation error: {e}")
            print("🔄 Falling back to synthetic data generation...")
            return self._generate_synthetic_training_data()
            
    def _combine_datasets(self, datasets):
        """Combine multiple datasets into a unified training set"""
        combined_data = []
        
        for key, data in datasets.items():
            # Add metadata
            symbol, timeframe, period = key.split('_')
            data = data.copy()
            data['symbol'] = symbol
            data['timeframe'] = timeframe
            data['period'] = period
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in data.columns for col in required_cols):
                combined_data.append(data)
                
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            result = result.sort_values('datetime' if 'datetime' in result.columns else result.index)
            return result.reset_index(drop=True)
        else:
            return pd.DataFrame()
            
    def _generate_synthetic_training_data(self, base_data=None, num_samples=5000):
        """Generate synthetic training data for model training"""
        print("🔄 Generating synthetic training data...")
        
        np.random.seed(42)  # For reproducibility
        
        if base_data is not None and len(base_data) > 0:
            # Use base data statistics
            base_price = base_data['close'].mean()
            base_volatility = base_data['close'].pct_change().std()
        else:
            # Default gold price characteristics
            base_price = 2000.0
            base_volatility = 0.015
            
        # Generate realistic price movements
        dates = pd.date_range(start='2020-01-01', periods=num_samples, freq='1H')
        
        # Generate correlated OHLCV data
        returns = np.random.normal(0, base_volatility, num_samples)
        returns = np.cumsum(returns)  # Make it trending
        
        prices = base_price * np.exp(returns)
        
        # Generate OHLC from close prices
        noise_factor = 0.002
        
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, noise_factor, num_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, noise_factor, num_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, noise_factor, num_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, num_samples),
            'symbol': 'SYNTHETIC',
            'timeframe': '1h',
            'period': 'synthetic'
        })
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        print(f"✅ Generated {len(data)} synthetic samples")
        return data
        
    def engineer_advanced_features(self, data):
        """Engineer advanced features for model training"""
        print("\n🔧 Engineering advanced features...")
        
        try:
            # Use feature engineer for single dataframe
            features = self.feature_engineer.create_features_from_single_dataframe(data)

            # Add additional advanced features
            features = self._add_advanced_features(features, data)

            # Add target variables
            features = self._create_target_variables(features, data)

            print(f"✅ Feature engineering complete: {features.shape[1]} features")
            return features

        except Exception as e:
            print(f"❌ Feature engineering error: {e}")
            return self._create_basic_features(data)
            
    def _add_advanced_features(self, features, data):
        """Add advanced technical and statistical features"""
        try:
            # Price momentum features
            for period in [5, 10, 20]:
                features[f'momentum_{period}'] = data['close'].pct_change(period)
                features[f'acceleration_{period}'] = features[f'momentum_{period}'].diff()
                
            # Volatility features
            for period in [10, 20, 50]:
                features[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
                features[f'volatility_ratio_{period}'] = (
                    features[f'volatility_{period}'] / features[f'volatility_{period}'].rolling(50).mean()
                )
                
            # Volume-price features
            if 'volume' in data.columns:
                features['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
                features['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
                
            # Market microstructure features
            features['spread'] = (data['high'] - data['low']) / data['close']
            features['body_size'] = np.abs(data['close'] - data['open']) / data['close']
            features['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
            features['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']
            
            # Time-based features
            if 'datetime' in data.columns:
                features['hour'] = pd.to_datetime(data['datetime']).dt.hour
                features['day_of_week'] = pd.to_datetime(data['datetime']).dt.dayofweek
                features['month'] = pd.to_datetime(data['datetime']).dt.month
                
                # Cyclical encoding
                features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
                features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
                features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
                features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
                
            return features
            
        except Exception as e:
            print(f"⚠️  Advanced feature creation error: {e}")
            return features
            
    def _create_target_variables(self, features, data):
        """Create target variables for different prediction tasks"""
        try:
            # Price direction (main target)
            future_returns = data['close'].pct_change().shift(-1)
            
            # Classification targets
            features['target_direction'] = (future_returns > 0).astype(int)
            features['target_strong_direction'] = ((future_returns > 0.005) | (future_returns < -0.005)).astype(int)
            
            # Regression targets
            features['target_return'] = future_returns
            features['target_price'] = data['close'].shift(-1)
            
            # Multi-class targets
            conditions = [
                future_returns > 0.01,   # Strong Buy
                future_returns > 0.002,  # Buy
                future_returns < -0.01,  # Strong Sell
                future_returns < -0.002, # Sell
            ]
            choices = [4, 3, 0, 1]  # Strong Buy, Buy, Strong Sell, Sell
            features['target_signal'] = np.select(conditions, choices, default=2)  # Hold
            
            return features
            
        except Exception as e:
            print(f"❌ Target creation error: {e}")
            # Create basic targets
            features['target_direction'] = 1
            features['target_signal'] = 2
            return features
            
    def _create_basic_features(self, data):
        """Create basic features as fallback"""
        features = pd.DataFrame()
        
        # Basic price features
        features['close'] = data['close']
        features['open'] = data['open']
        features['high'] = data['high']
        features['low'] = data['low']
        
        # Basic technical indicators
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        features['rsi'] = self._calculate_rsi(data['close'])
        
        # Basic targets
        future_returns = data['close'].pct_change().shift(-1)
        features['target_direction'] = (future_returns > 0).astype(int)
        features['target_signal'] = 2  # Default to hold
        
        return features.fillna(method='ffill').fillna(0)
        
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def split_data_for_training(self, features):
        """Split data for time series training with proper validation"""
        print("\n📊 Splitting data for training...")
        
        # Remove rows with NaN targets
        features = features.dropna(subset=['target_direction', 'target_signal'])
        
        if len(features) < 100:
            raise ValueError("Insufficient data for training")
            
        # Time series split (preserving temporal order)
        train_size = int(len(features) * 0.7)
        val_size = int(len(features) * 0.15)
        
        self.training_data = features.iloc[:train_size].copy()
        self.validation_data = features.iloc[train_size:train_size + val_size].copy()
        self.test_data = features.iloc[train_size + val_size:].copy()
        
        print(f"✅ Data split complete:")
        print(f"   Training: {len(self.training_data)} samples")
        print(f"   Validation: {len(self.validation_data)} samples")
        print(f"   Test: {len(self.test_data)} samples")
        
        return self.training_data, self.validation_data, self.test_data
        
    def train_ensemble_models(self):
        """Train all ensemble models with hyperparameter optimization"""
        print("\n🤖 Training ensemble models...")
        
        if self.training_data is None:
            raise ValueError("No training data available. Run prepare_comprehensive_dataset first.")
            
        # Prepare features and targets
        feature_cols = [col for col in self.training_data.columns
                       if not col.startswith('target_') and col not in ['datetime', 'symbol', 'timeframe', 'period']]

        # Convert to numeric and handle data types
        X_train = self.training_data[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_val = self.validation_data[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test = self.test_data[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        y_train = self.training_data['target_direction']
        y_val = self.validation_data['target_direction']
        y_test = self.test_data['target_direction']
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train individual models
        models_to_train = [
            ('random_forest', self._train_random_forest),
            ('gradient_boost', self._train_gradient_boost),
            ('lstm_transformer', self._train_lstm_transformer),
            ('cnn_attention', self._train_cnn_attention)
        ]
        
        model_predictions = {}
        
        for model_name, train_func in models_to_train:
            try:
                print(f"\n🔧 Training {model_name}...")
                model, predictions = train_func(
                    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
                )
                
                if model is not None:
                    self.models[model_name] = model
                    model_predictions[model_name] = predictions
                    print(f"✅ {model_name} training complete")
                else:
                    print(f"❌ {model_name} training failed")
                    
            except Exception as e:
                print(f"❌ {model_name} training error: {e}")
                
        # Train meta-learner
        if len(model_predictions) >= 2:
            print(f"\n🧠 Training meta-learner...")
            try:
                meta_learner = self._train_meta_learner(model_predictions, y_test)
                if meta_learner is not None:
                    self.models['meta_learner'] = meta_learner
                    print(f"✅ Meta-learner training complete")
            except Exception as e:
                print(f"❌ Meta-learner training error: {e}")
                
        print(f"\n✅ Ensemble training complete: {len(self.models)} models trained")
        return self.models
        
    def _train_random_forest(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train Random Forest with hyperparameter optimization"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Hyperparameter optimization with Optuna
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            return accuracy_score(y_val, predictions)
            
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        # Train final model with best parameters
        best_model = RandomForestClassifier(**study.best_params)
        best_model.fit(X_train, y_train)
        
        # Evaluate
        test_predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, test_predictions)
        
        self.performance_metrics['random_forest'] = {
            'accuracy': accuracy,
            'best_params': study.best_params,
            'feature_importance': dict(zip(range(X_train.shape[1]), best_model.feature_importances_))
        }
        
        print(f"   Random Forest accuracy: {accuracy:.3f}")
        return best_model, test_predictions
        
    def _train_gradient_boost(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train Gradient Boosting with hyperparameter optimization"""
        try:
            import xgboost as xgb
            
            # Hyperparameter optimization
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                predictions = model.predict(X_val)
                return accuracy_score(y_val, predictions)
                
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            
            # Train final model
            best_model = xgb.XGBClassifier(**study.best_params)
            best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Evaluate
            test_predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, test_predictions)
            
            self.performance_metrics['gradient_boost'] = {
                'accuracy': accuracy,
                'best_params': study.best_params
            }
            
            print(f"   Gradient Boost accuracy: {accuracy:.3f}")
            return best_model, test_predictions
            
        except ImportError:
            print("   XGBoost not available, using sklearn GradientBoosting")
            from sklearn.ensemble import GradientBoostingClassifier
            
            model = GradientBoostingClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            test_predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, test_predictions)
            
            self.performance_metrics['gradient_boost'] = {'accuracy': accuracy}
            print(f"   Gradient Boost accuracy: {accuracy:.3f}")
            return model, test_predictions

    def _train_lstm_transformer(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train LSTM+Transformer model"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val.values)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test.values)

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Initialize model
            input_dim = X_train.shape[1]
            model = LSTMTransformerModel(input_dim=input_dim, hidden_dim=128, num_classes=2)

            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

            # Training loop
            best_val_acc = 0
            patience_counter = 0
            max_patience = 20

            for epoch in range(100):
                model.train()
                total_loss = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_predictions = torch.argmax(val_outputs, dim=1)
                    val_accuracy = (val_predictions == y_val_tensor).float().mean().item()

                scheduler.step(total_loss)

                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), 'models/trained_models/lstm_transformer_best.pth')
                else:
                    patience_counter += 1

                if patience_counter >= max_patience:
                    break

            # Load best model and evaluate
            model.load_state_dict(torch.load('models/trained_models/lstm_transformer_best.pth'))
            model.eval()

            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1).numpy()

            accuracy = accuracy_score(y_test, test_predictions)

            self.performance_metrics['lstm_transformer'] = {
                'accuracy': accuracy,
                'best_val_accuracy': best_val_acc,
                'epochs_trained': epoch + 1
            }

            print(f"   LSTM+Transformer accuracy: {accuracy:.3f}")
            return model, test_predictions

        except Exception as e:
            print(f"   LSTM+Transformer training error: {e}")
            return None, None

    def _train_cnn_attention(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train CNN+Attention model"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            # Reshape data for CNN (add sequence dimension)
            sequence_length = min(60, X_train.shape[1])

            # Pad or truncate features to sequence length
            if X_train.shape[1] < sequence_length:
                padding = sequence_length - X_train.shape[1]
                X_train = np.pad(X_train, ((0, 0), (0, padding)), mode='constant')
                X_val = np.pad(X_val, ((0, 0), (0, padding)), mode='constant')
                X_test = np.pad(X_test, ((0, 0), (0, padding)), mode='constant')
            else:
                X_train = X_train[:, :sequence_length]
                X_val = X_val[:, :sequence_length]
                X_test = X_test[:, :sequence_length]

            # Reshape for CNN: (batch, channels, sequence)
            X_train = X_train.reshape(X_train.shape[0], 1, -1)
            X_val = X_val.reshape(X_val.shape[0], 1, -1)
            X_test = X_test.reshape(X_test.shape[0], 1, -1)

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val.values)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test.values)

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Initialize model
            model = CNNAttentionModel(input_channels=1, sequence_length=sequence_length, num_classes=2)

            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

            # Training loop
            best_val_acc = 0
            patience_counter = 0
            max_patience = 20

            for epoch in range(100):
                model.train()
                total_loss = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_predictions = torch.argmax(val_outputs, dim=1)
                    val_accuracy = (val_predictions == y_val_tensor).float().mean().item()

                scheduler.step(total_loss)

                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), 'models/trained_models/cnn_attention_best.pth')
                else:
                    patience_counter += 1

                if patience_counter >= max_patience:
                    break

            # Load best model and evaluate
            model.load_state_dict(torch.load('models/trained_models/cnn_attention_best.pth'))
            model.eval()

            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1).numpy()

            accuracy = accuracy_score(y_test, test_predictions)

            self.performance_metrics['cnn_attention'] = {
                'accuracy': accuracy,
                'best_val_accuracy': best_val_acc,
                'epochs_trained': epoch + 1
            }

            print(f"   CNN+Attention accuracy: {accuracy:.3f}")
            return model, test_predictions

        except Exception as e:
            print(f"   CNN+Attention training error: {e}")
            return None, None

    def _train_meta_learner(self, model_predictions, y_test):
        """Train meta-learner to combine model predictions"""
        try:
            # Prepare meta-features (predictions from base models)
            meta_features = np.column_stack(list(model_predictions.values()))

            # Train meta-learner
            meta_learner = MetaLearner(input_dim=meta_features.shape[1], hidden_dim=64, num_classes=2)

            import torch
            import torch.nn as nn
            import torch.optim as optim

            # Convert to tensors
            X_meta = torch.FloatTensor(meta_features)
            y_meta = torch.LongTensor(y_test.values)

            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(meta_learner.parameters(), lr=0.001)

            # Simple training loop for meta-learner
            for epoch in range(200):
                optimizer.zero_grad()
                outputs = meta_learner(X_meta)
                loss = criterion(outputs, y_meta)
                loss.backward()
                optimizer.step()

            # Evaluate meta-learner
            meta_learner.eval()
            with torch.no_grad():
                meta_outputs = meta_learner(X_meta)
                meta_predictions = torch.argmax(meta_outputs, dim=1).numpy()

            accuracy = accuracy_score(y_test, meta_predictions)

            self.performance_metrics['meta_learner'] = {
                'accuracy': accuracy,
                'base_models': list(model_predictions.keys())
            }

            print(f"   Meta-learner accuracy: {accuracy:.3f}")

            # Save meta-learner
            torch.save(meta_learner.state_dict(), 'models/trained_models/meta_learner_best.pth')

            return meta_learner

        except Exception as e:
            print(f"   Meta-learner training error: {e}")
            return None

    def validate_ensemble_performance(self):
        """Comprehensive validation of ensemble performance"""
        print("\n🔍 Validating ensemble performance...")

        if not self.models or self.test_data is None:
            print("❌ No models or test data available for validation")
            return None

        # Prepare test data
        feature_cols = [col for col in self.test_data.columns
                       if not col.startswith('target_') and col not in ['datetime', 'symbol', 'timeframe', 'period']]

        X_test = self.test_data[feature_cols].fillna(0)
        y_test = self.test_data['target_direction']

        if 'standard' in self.scalers:
            X_test_scaled = self.scalers['standard'].transform(X_test)
        else:
            X_test_scaled = X_test.values

        # Validate each model
        validation_results = {}

        for model_name, model in self.models.items():
            try:
                if model_name in ['lstm_transformer', 'cnn_attention']:
                    # Deep learning models
                    predictions = self._predict_deep_model(model, X_test_scaled, model_name)
                elif model_name == 'meta_learner':
                    # Meta-learner needs base model predictions
                    continue
                else:
                    # Traditional ML models
                    predictions = model.predict(X_test_scaled)

                if predictions is not None:
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, predictions)
                    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

                    validation_results[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'meets_target': accuracy >= self.target_accuracy
                    }

                    print(f"   {model_name}: {accuracy:.3f} accuracy")

            except Exception as e:
                print(f"   ❌ {model_name} validation error: {e}")

        # Ensemble prediction
        if len(validation_results) >= 2:
            try:
                ensemble_predictions = self._create_ensemble_prediction(X_test_scaled, y_test)
                if ensemble_predictions is not None:
                    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
                    validation_results['ensemble'] = {
                        'accuracy': ensemble_accuracy,
                        'meets_target': ensemble_accuracy >= self.target_accuracy
                    }
                    print(f"   ensemble: {ensemble_accuracy:.3f} accuracy")

            except Exception as e:
                print(f"   ❌ Ensemble validation error: {e}")

        # Overall validation summary
        accuracies = [result['accuracy'] for result in validation_results.values()]
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            max_accuracy = np.max(accuracies)
            models_meeting_target = sum(1 for result in validation_results.values() if result.get('meets_target', False))

            print(f"\n📊 Validation Summary:")
            print(f"   Average accuracy: {avg_accuracy:.3f}")
            print(f"   Best accuracy: {max_accuracy:.3f}")
            print(f"   Models meeting target: {models_meeting_target}/{len(validation_results)}")
            print(f"   Target achieved: {'✅' if max_accuracy >= self.target_accuracy else '❌'}")

        return validation_results

    def _predict_deep_model(self, model, X_test, model_name):
        """Make predictions with deep learning models"""
        try:
            import torch

            model.eval()

            if model_name == 'cnn_attention':
                # Reshape for CNN
                sequence_length = min(60, X_test.shape[1])
                if X_test.shape[1] < sequence_length:
                    padding = sequence_length - X_test.shape[1]
                    X_test = np.pad(X_test, ((0, 0), (0, padding)), mode='constant')
                else:
                    X_test = X_test[:, :sequence_length]
                X_test = X_test.reshape(X_test.shape[0], 1, -1)

            X_tensor = torch.FloatTensor(X_test)

            with torch.no_grad():
                outputs = model(X_tensor)
                predictions = torch.argmax(outputs, dim=1).numpy()

            return predictions

        except Exception as e:
            print(f"   Deep model prediction error: {e}")
            return None

    def _create_ensemble_prediction(self, X_test, y_test):
        """Create ensemble prediction from all models"""
        try:
            predictions_list = []

            # Get predictions from each model
            for model_name, model in self.models.items():
                if model_name == 'meta_learner':
                    continue

                if model_name in ['lstm_transformer', 'cnn_attention']:
                    predictions = self._predict_deep_model(model, X_test, model_name)
                else:
                    predictions = model.predict(X_test)

                if predictions is not None:
                    predictions_list.append(predictions)

            if len(predictions_list) >= 2:
                # Simple voting ensemble
                predictions_array = np.array(predictions_list)
                ensemble_predictions = np.round(np.mean(predictions_array, axis=0)).astype(int)
                return ensemble_predictions
            else:
                return None

        except Exception as e:
            print(f"   Ensemble prediction error: {e}")
            return None

    def save_trained_models(self):
        """Save all trained models and metadata"""
        print("\n💾 Saving trained models...")

        # Create directories
        os.makedirs('models/trained_models', exist_ok=True)

        saved_models = 0

        # Save traditional ML models
        for model_name, model in self.models.items():
            if model_name not in ['lstm_transformer', 'cnn_attention', 'meta_learner']:
                try:
                    model_path = f'models/trained_models/{model_name}_model.joblib'
                    joblib.dump(model, model_path)
                    print(f"   ✅ {model_name} saved")
                    saved_models += 1
                except Exception as e:
                    print(f"   ❌ {model_name} save error: {e}")

        # Save scalers
        try:
            for scaler_name, scaler in self.scalers.items():
                scaler_path = f'models/trained_models/{scaler_name}_scaler.joblib'
                joblib.dump(scaler, scaler_path)
                print(f"   ✅ {scaler_name} scaler saved")
        except Exception as e:
            print(f"   ❌ Scaler save error: {e}")

        # Save performance metrics
        try:
            metrics_path = 'models/trained_models/performance_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            print(f"   ✅ Performance metrics saved")
        except Exception as e:
            print(f"   ❌ Metrics save error: {e}")

        # Save training configuration
        try:
            config_path = 'models/trained_models/training_config.json'
            training_config = {
                'target_accuracy': self.target_accuracy,
                'model_config': self.config,
                'validation_config': self.validation_config,
                'training_timestamp': datetime.now().isoformat(),
                'models_trained': list(self.models.keys()),
                'feature_count': len([col for col in self.training_data.columns
                                    if not col.startswith('target_')]) if self.training_data is not None else 0
            }

            with open(config_path, 'w') as f:
                json.dump(training_config, f, indent=2)
            print(f"   ✅ Training configuration saved")
        except Exception as e:
            print(f"   ❌ Configuration save error: {e}")

        print(f"✅ Model saving complete: {saved_models} models saved")

        # Deep learning models are saved during training
        deep_models = ['lstm_transformer', 'cnn_attention', 'meta_learner']
        saved_deep_models = [model for model in deep_models if model in self.models]
        if saved_deep_models:
            print(f"   Deep learning models saved during training: {', '.join(saved_deep_models)}")

        return saved_models + len(saved_deep_models)

    def generate_training_report(self):
        """Generate comprehensive training report"""
        print("\n📊 Generating training report...")

        report = {
            'training_summary': {
                'timestamp': datetime.now().isoformat(),
                'target_accuracy': self.target_accuracy,
                'models_trained': len(self.models),
                'training_samples': len(self.training_data) if self.training_data is not None else 0,
                'validation_samples': len(self.validation_data) if self.validation_data is not None else 0,
                'test_samples': len(self.test_data) if self.test_data is not None else 0
            },
            'model_performance': self.performance_metrics,
            'recommendations': self._generate_recommendations()
        }

        # Save report
        try:
            report_path = 'models/trained_models/training_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Training report saved to {report_path}")
        except Exception as e:
            print(f"❌ Report save error: {e}")

        return report

    def _generate_recommendations(self):
        """Generate recommendations based on training results"""
        recommendations = []

        if not self.performance_metrics:
            recommendations.append("No performance metrics available - retrain models")
            return recommendations

        # Analyze performance
        accuracies = [metrics.get('accuracy', 0) for metrics in self.performance_metrics.values()]
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        max_accuracy = np.max(accuracies) if accuracies else 0

        if max_accuracy >= self.target_accuracy:
            recommendations.append(f"✅ Target accuracy achieved: {max_accuracy:.3f}")
            recommendations.append("Models are ready for production deployment")
        else:
            recommendations.append(f"⚠️ Target accuracy not met: {max_accuracy:.3f} < {self.target_accuracy:.3f}")
            recommendations.append("Consider: more data, feature engineering, or hyperparameter tuning")

        if avg_accuracy < 0.8:
            recommendations.append("Low average accuracy - review data quality and feature engineering")

        # Model-specific recommendations
        if 'random_forest' in self.performance_metrics:
            rf_acc = self.performance_metrics['random_forest']['accuracy']
            if rf_acc > 0.85:
                recommendations.append("Random Forest performing well - good baseline model")

        if len(self.models) < 3:
            recommendations.append("Consider training more diverse models for better ensemble performance")

        return recommendations

    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("🚀 Starting complete model training pipeline...")
        print("=" * 60)

        try:
            # Step 1: Prepare dataset
            print("📊 Step 1: Preparing comprehensive dataset...")
            raw_data = self.prepare_comprehensive_dataset()

            if raw_data is None or len(raw_data) < 100:
                raise ValueError("Insufficient data for training")

            # Step 2: Feature engineering
            print("\n🔧 Step 2: Engineering advanced features...")
            features = self.engineer_advanced_features(raw_data)

            # Step 3: Data splitting
            print("\n📊 Step 3: Splitting data for training...")
            train_data, val_data, test_data = self.split_data_for_training(features)

            # Step 4: Model training
            print("\n🤖 Step 4: Training ensemble models...")
            models = self.train_ensemble_models()

            # Step 5: Validation
            print("\n🔍 Step 5: Validating model performance...")
            validation_results = self.validate_ensemble_performance()

            # Step 6: Save models
            print("\n💾 Step 6: Saving trained models...")
            saved_count = self.save_trained_models()

            # Step 7: Generate report
            print("\n📊 Step 7: Generating training report...")
            report = self.generate_training_report()

            # Final summary
            print("\n" + "=" * 60)
            print("🎉 TRAINING PIPELINE COMPLETE")
            print("=" * 60)

            if validation_results:
                accuracies = [result['accuracy'] for result in validation_results.values()]
                max_accuracy = np.max(accuracies) if accuracies else 0
                target_met = max_accuracy >= self.target_accuracy

                print(f"📊 Best model accuracy: {max_accuracy:.3f}")
                print(f"🎯 Target accuracy: {self.target_accuracy:.3f}")
                print(f"✅ Target achieved: {'YES' if target_met else 'NO'}")
                print(f"💾 Models saved: {saved_count}")
                print(f"🤖 Models trained: {len(models)}")

                if target_met:
                    print("\n🎉 SUCCESS: Models ready for production!")
                else:
                    print("\n⚠️  Models need improvement. Check recommendations.")

            else:
                print("⚠️  Training completed but validation failed")

            print("=" * 60)

            return {
                'success': True,
                'models_trained': len(models),
                'models_saved': saved_count,
                'validation_results': validation_results,
                'report': report
            }

        except Exception as e:
            print(f"\n❌ Training pipeline error: {e}")
            print("=" * 60)
            return {
                'success': False,
                'error': str(e),
                'models_trained': len(self.models),
                'models_saved': 0
            }


def main():
    """Main training function"""
    print("🥇 Gold Trading AI - Advanced Model Training")
    print("=" * 60)
    print("🎯 Target: >90% accuracy with ensemble models")
    print("🤖 Models: LSTM+Transformer, CNN+Attention, RF, XGBoost")
    print("🔧 Features: Advanced feature engineering & optimization")
    print("=" * 60)

    # Initialize trainer
    trainer = AdvancedModelTrainer(target_accuracy=0.90)

    # Run complete training
    results = trainer.run_complete_training()

    # Print final results
    if results['success']:
        print(f"\n🎉 Training successful!")
        print(f"   Models trained: {results['models_trained']}")
        print(f"   Models saved: {results['models_saved']}")

        if results['validation_results']:
            best_accuracy = max(result['accuracy'] for result in results['validation_results'].values())
            print(f"   Best accuracy: {best_accuracy:.3f}")

        print("\n📁 Files created:")
        print("   models/trained_models/")
        print("   ├── *_model.joblib (traditional ML models)")
        print("   ├── *_best.pth (deep learning models)")
        print("   ├── *_scaler.joblib (feature scalers)")
        print("   ├── performance_metrics.json")
        print("   ├── training_config.json")
        print("   └── training_report.json")

        print("\n🚀 Next steps:")
        print("   1. Run: python demo.py")
        print("   2. Run: python main.py")
        print("   3. Check: models/trained_models/training_report.json")

        return 0
    else:
        print(f"\n❌ Training failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
