#!/usr/bin/env python3
"""
Gold Trading AI - Quick Model Training
Fast training with optimized models for immediate results

Author: AI Trading Systems
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.advanced_data_fetcher import AdvancedDataFetcher
from utils.feature_engineering import FeatureEngineer


class QuickTrainer:
    """Quick model trainer for immediate results"""
    
    def __init__(self):
        self.data_fetcher = AdvancedDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
        print("🚀 Quick Trainer initialized")
        
    def prepare_data(self):
        """Prepare training data quickly"""
        print("\n📊 Preparing training data...")
        
        try:
            # Fetch recent gold data
            print("   📈 Fetching GC=F data...")
            data = self.data_fetcher.fetch_historical_data('GC=F', '1y', '1h')
            
            if data is None or len(data) < 1000:
                print("   📈 Trying GLD...")
                data = self.data_fetcher.fetch_historical_data('GLD', '1y', '1h')
                
            if data is None or len(data) < 1000:
                print("   🔄 Generating synthetic data...")
                data = self._generate_synthetic_data()
                
            print(f"   ✅ Dataset ready: {len(data)} records")
            return data
            
        except Exception as e:
            print(f"   ❌ Data preparation error: {e}")
            return self._generate_synthetic_data()
            
    def _generate_synthetic_data(self, num_samples=5000):
        """Generate synthetic gold price data"""
        np.random.seed(42)
        
        # Generate realistic gold price movements
        dates = pd.date_range(start='2023-01-01', periods=num_samples, freq='1H')
        
        # Base gold price around $2000
        base_price = 2000.0
        volatility = 0.015
        
        # Generate price series with trend and noise
        returns = np.random.normal(0, volatility, num_samples)
        returns = np.cumsum(returns)
        prices = base_price * np.exp(returns)
        
        # Generate OHLCV data
        noise = 0.002
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, noise, num_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, noise, num_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, noise, num_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, num_samples)
        })
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data
        
    def create_features(self, data):
        """Create features quickly"""
        print("\n🔧 Creating features...")
        
        try:
            # Use feature engineer
            features = self.feature_engineer.create_features_from_single_dataframe(data)
            
            # Add simple target
            future_returns = features['close'].pct_change().shift(-1)
            features['target'] = (future_returns > 0).astype(int)
            
            # Clean data
            features = features.dropna()
            
            print(f"   ✅ Features created: {features.shape[1]} columns, {len(features)} rows")
            return features
            
        except Exception as e:
            print(f"   ❌ Feature creation error: {e}")
            return self._create_basic_features(data)
            
    def _create_basic_features(self, data):
        """Create basic features as fallback"""
        features = pd.DataFrame()
        
        # Basic price features
        features['close'] = pd.to_numeric(data['close'], errors='coerce')
        features['open'] = pd.to_numeric(data['open'], errors='coerce')
        features['high'] = pd.to_numeric(data['high'], errors='coerce')
        features['low'] = pd.to_numeric(data['low'], errors='coerce')
        
        # Simple technical indicators
        features['sma_20'] = features['close'].rolling(20).mean()
        features['sma_50'] = features['close'].rolling(50).mean()
        features['price_change'] = features['close'].pct_change()
        features['volatility'] = features['price_change'].rolling(20).std()
        
        # Simple RSI
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Target
        future_returns = features['close'].pct_change().shift(-1)
        features['target'] = (future_returns > 0).astype(int)
        
        # Clean
        features = features.fillna(method='ffill').fillna(0)
        features = features.dropna()
        
        return features
        
    def train_models(self, features):
        """Train models quickly"""
        print("\n🤖 Training models...")
        
        # Prepare data
        feature_cols = [col for col in features.columns if col != 'target']
        X = features[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = features['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X_train.shape[1]}")
        
        # Train models
        models_to_train = [
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('XGBoost', xgb.XGBClassifier(n_estimators=100, random_state=42)),
            ('LightGBM', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        ]
        
        for name, model in models_to_train:
            try:
                print(f"\n   🔧 Training {name}...")
                
                if name in ['Random Forest', 'Gradient Boosting']:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, predictions)
                
                self.models[name] = model
                self.results[name] = {
                    'accuracy': accuracy,
                    'predictions': predictions
                }
                
                print(f"      ✅ {name}: {accuracy:.3f} accuracy")
                
            except Exception as e:
                print(f"      ❌ {name} error: {e}")
                
        return X_test, y_test
        
    def create_ensemble(self, X_test, y_test):
        """Create ensemble prediction"""
        print("\n🧠 Creating ensemble...")
        
        if len(self.models) < 2:
            print("   ⚠️  Not enough models for ensemble")
            return
            
        try:
            # Get predictions from all models
            predictions_list = []
            
            for name, model in self.models.items():
                if name in ['Random Forest', 'Gradient Boosting']:
                    X_test_scaled = self.scaler.transform(X_test)
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                predictions_list.append(pred)
                
            # Simple voting ensemble
            ensemble_pred = np.round(np.mean(predictions_list, axis=0)).astype(int)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            
            self.results['Ensemble'] = {
                'accuracy': ensemble_accuracy,
                'predictions': ensemble_pred
            }
            
            print(f"   ✅ Ensemble: {ensemble_accuracy:.3f} accuracy")
            
        except Exception as e:
            print(f"   ❌ Ensemble error: {e}")
            
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving models...")
        
        os.makedirs('models/trained_models', exist_ok=True)
        
        saved_count = 0
        
        # Save models
        for name, model in self.models.items():
            try:
                filename = f"models/trained_models/{name.lower().replace(' ', '_')}_quick.joblib"
                joblib.dump(model, filename)
                print(f"   ✅ {name} saved")
                saved_count += 1
            except Exception as e:
                print(f"   ❌ {name} save error: {e}")
                
        # Save scaler
        try:
            joblib.dump(self.scaler, 'models/trained_models/scaler_quick.joblib')
            print(f"   ✅ Scaler saved")
        except Exception as e:
            print(f"   ❌ Scaler save error: {e}")
            
        # Save results
        try:
            results_to_save = {}
            for name, result in self.results.items():
                results_to_save[name] = {
                    'accuracy': result['accuracy']
                }
                
            with open('models/trained_models/quick_results.json', 'w') as f:
                json.dump(results_to_save, f, indent=2)
            print(f"   ✅ Results saved")
        except Exception as e:
            print(f"   ❌ Results save error: {e}")
            
        return saved_count
        
    def run_quick_training(self):
        """Run complete quick training"""
        print("🚀 Starting quick training pipeline...")
        print("=" * 50)
        
        try:
            # Prepare data
            data = self.prepare_data()
            
            # Create features
            features = self.create_features(data)
            
            # Train models
            X_test, y_test = self.train_models(features)
            
            # Create ensemble
            self.create_ensemble(X_test, y_test)
            
            # Save models
            saved_count = self.save_models()
            
            # Summary
            print("\n" + "=" * 50)
            print("🎉 QUICK TRAINING COMPLETE")
            print("=" * 50)
            
            if self.results:
                best_accuracy = max(result['accuracy'] for result in self.results.values())
                print(f"📊 Best accuracy: {best_accuracy:.3f}")
                print(f"🤖 Models trained: {len(self.models)}")
                print(f"💾 Models saved: {saved_count}")
                
                print(f"\n📊 Model Performance:")
                for name, result in self.results.items():
                    accuracy = result['accuracy']
                    status = "✅" if accuracy >= 0.85 else "⚠️" if accuracy >= 0.75 else "❌"
                    print(f"   {status} {name}: {accuracy:.3f}")
                    
                if best_accuracy >= 0.90:
                    print(f"\n🎯 TARGET ACHIEVED: >90% accuracy!")
                elif best_accuracy >= 0.85:
                    print(f"\n✅ EXCELLENT: >85% accuracy achieved")
                elif best_accuracy >= 0.75:
                    print(f"\n👍 GOOD: >75% accuracy achieved")
                else:
                    print(f"\n⚠️  NEEDS IMPROVEMENT: <75% accuracy")
                    
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Quick training error: {e}")
            return False


def main():
    """Main function"""
    print("🥇 Gold Trading AI - Quick Model Training")
    print("🎯 Fast training for immediate results")
    print("=" * 50)
    
    trainer = QuickTrainer()
    success = trainer.run_quick_training()
    
    if success:
        print("\n🚀 Quick training successful!")
        print("📁 Models saved to: models/trained_models/")
        print("🔄 Run: python demo.py")
    else:
        print("\n❌ Quick training failed")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
