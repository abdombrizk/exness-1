#!/usr/bin/env python3
"""
Feature Aligner for Gold Trading AI
Ensures feature compatibility between training and prediction

Author: AI Trading Systems
Version: 1.0.0
"""

import os
import json
import numpy as np
import pandas as pd


class FeatureAligner:
    """Aligns features between training and prediction phases"""
    
    def __init__(self):
        self.training_features = None
        self.feature_mapping = {}
        self.default_values = {}
        
    def save_training_features(self, features, save_path="models/trained_models/feature_schema.json"):
        """Save the feature schema from training"""
        try:
            # Get feature names and types
            feature_info = {}
            
            for col in features.columns:
                if not col.startswith('target_'):
                    feature_info[col] = {
                        'dtype': str(features[col].dtype),
                        'mean': float(features[col].mean()) if pd.api.types.is_numeric_dtype(features[col]) else 0.0,
                        'std': float(features[col].std()) if pd.api.types.is_numeric_dtype(features[col]) else 1.0,
                        'min': float(features[col].min()) if pd.api.types.is_numeric_dtype(features[col]) else 0.0,
                        'max': float(features[col].max()) if pd.api.types.is_numeric_dtype(features[col]) else 1.0
                    }
                    
            schema = {
                'feature_names': list(feature_info.keys()),
                'feature_count': len(feature_info),
                'feature_info': feature_info,
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            # Save schema
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(schema, f, indent=2)
                
            print(f"✅ Feature schema saved: {len(feature_info)} features")
            return True
            
        except Exception as e:
            print(f"❌ Feature schema save error: {e}")
            return False
            
    def load_training_features(self, load_path="models/trained_models/feature_schema.json"):
        """Load the feature schema from training"""
        try:
            if not os.path.exists(load_path):
                print(f"⚠️  Feature schema not found: {load_path}")
                return False
                
            with open(load_path, 'r') as f:
                schema = json.load(f)
                
            self.training_features = schema['feature_names']
            self.feature_mapping = schema['feature_info']
            
            # Set default values
            for feature, info in self.feature_mapping.items():
                self.default_values[feature] = info.get('mean', 0.0)
                
            print(f"✅ Feature schema loaded: {len(self.training_features)} features")
            return True
            
        except Exception as e:
            print(f"❌ Feature schema load error: {e}")
            return False
            
    def align_features(self, prediction_features):
        """Align prediction features to match training features"""
        try:
            if self.training_features is None:
                print("⚠️  No training feature schema loaded")
                return prediction_features
                
            # Convert to DataFrame if needed
            if isinstance(prediction_features, dict):
                prediction_features = pd.DataFrame([prediction_features])
            elif isinstance(prediction_features, (list, np.ndarray)):
                prediction_features = pd.DataFrame(prediction_features)
                
            aligned_features = pd.DataFrame()
            
            # Add each training feature
            for feature in self.training_features:
                if feature in prediction_features.columns:
                    # Feature exists, use it
                    aligned_features[feature] = prediction_features[feature]
                else:
                    # Feature missing, use default value
                    default_val = self.default_values.get(feature, 0.0)
                    aligned_features[feature] = default_val
                    
            # Ensure correct data types
            for feature in aligned_features.columns:
                if feature in self.feature_mapping:
                    expected_dtype = self.feature_mapping[feature]['dtype']
                    if 'float' in expected_dtype:
                        aligned_features[feature] = pd.to_numeric(aligned_features[feature], errors='coerce').fillna(0.0)
                    elif 'int' in expected_dtype:
                        aligned_features[feature] = pd.to_numeric(aligned_features[feature], errors='coerce').fillna(0).astype(int)
                        
            print(f"✅ Features aligned: {len(aligned_features.columns)} features")
            return aligned_features
            
        except Exception as e:
            print(f"❌ Feature alignment error: {e}")
            return prediction_features
            
    def create_feature_schema_from_enhanced_training(self):
        """Create feature schema based on enhanced training results"""
        try:
            # Load enhanced training results to understand feature structure
            results_path = "models/trained_models/enhanced_results.json"
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    
            # Create a comprehensive feature list based on our feature engineering
            comprehensive_features = [
                # Basic OHLCV
                'open', 'high', 'low', 'close', 'volume',
                
                # Moving averages
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
                'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
                
                # Price ratios
                'price_sma20_ratio', 'sma20_sma50_ratio',
                
                # Momentum indicators
                'rsi_14', 'rsi_21', 'roc_10', 'roc_20',
                
                # MACD
                'macd', 'macd_signal', 'macd_histogram',
                
                # Bollinger Bands
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                
                # Stochastic
                'stoch_k', 'stoch_d',
                
                # ATR and volatility
                'atr_14', 'atr_20',
                
                # Williams %R and CCI
                'williams_r', 'cci',
                
                # ADX
                'adx', 'plus_di', 'minus_di',
                
                # Volume indicators
                'obv', 'ad', 'adosc',
                
                # Price patterns
                'price_change', 'price_change_pct', 'high_low_range', 'high_low_range_pct',
                
                # Candlestick patterns
                'doji', 'hammer', 'hanging_man', 'shooting_star', 'engulfing',
                
                # Momentum features
                'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
                'price_change_3', 'price_change_5', 'price_change_10', 'price_change_20',
                
                # Support/Resistance
                'resistance_level', 'support_level', 'price_to_resistance', 'price_to_support',
                
                # Trend features
                'higher_highs', 'lower_lows',
                
                # Volume features
                'volume_sma_10', 'volume_sma_20', 'volume_ratio', 'vwap', 'price_vwap_ratio',
                'volume_oscillator', 'money_flow_20',
                
                # Statistical features
                'volatility_10', 'volatility_20', 'volatility_50',
                'skewness_10', 'skewness_20', 'skewness_50',
                'kurtosis_10', 'kurtosis_20', 'kurtosis_50',
                'zscore_20', 'percentile_rank_50',
                'autocorr_1', 'autocorr_5',
                
                # Time features
                'hour', 'day_of_week', 'month', 'quarter',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
                'is_market_open', 'is_weekend',
                
                # Market regime features
                'trend_short', 'trend_medium', 'trend_strength',
                'vol_regime_low', 'vol_regime_medium', 'vol_regime_high',
                'bull_market', 'bear_market',
                
                # Mathematical features
                'fib_23.6', 'fib_38.2', 'fib_50.0', 'fib_61.8', 'dist_to_fib_50',
                'hurst_50', 'entropy_20',
                
                # Lag features
                'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
                
                # Rolling statistics
                'return_mean_5', 'return_mean_10', 'return_mean_20',
                'return_std_5', 'return_std_10', 'return_std_20',
                'return_skew_5', 'return_skew_10', 'return_skew_20'
            ]
            
            # Create feature info with default values
            feature_info = {}
            for feature in comprehensive_features:
                feature_info[feature] = {
                    'dtype': 'float64',
                    'mean': 0.0,
                    'std': 1.0,
                    'min': -10.0,
                    'max': 10.0
                }
                
            # Special handling for specific features
            price_features = ['open', 'high', 'low', 'close', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
                            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'bb_upper', 'bb_middle', 'bb_lower',
                            'resistance_level', 'support_level', 'vwap']
            
            for feature in price_features:
                if feature in feature_info:
                    feature_info[feature].update({
                        'mean': 2000.0,  # Typical gold price
                        'std': 100.0,
                        'min': 1500.0,
                        'max': 2500.0
                    })
                    
            volume_features = ['volume', 'obv', 'ad', 'volume_sma_10', 'volume_sma_20', 'money_flow_20']
            for feature in volume_features:
                if feature in feature_info:
                    feature_info[feature].update({
                        'mean': 50000.0,
                        'std': 25000.0,
                        'min': 0.0,
                        'max': 200000.0
                    })
                    
            schema = {
                'feature_names': comprehensive_features,
                'feature_count': len(comprehensive_features),
                'feature_info': feature_info,
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            # Save schema
            schema_path = "models/trained_models/feature_schema.json"
            os.makedirs(os.path.dirname(schema_path), exist_ok=True)
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
                
            print(f"✅ Comprehensive feature schema created: {len(comprehensive_features)} features")
            
            # Load it
            self.load_training_features(schema_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Feature schema creation error: {e}")
            return False


# Global instance
feature_aligner = FeatureAligner()
