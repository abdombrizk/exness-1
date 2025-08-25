#!/usr/bin/env python3
"""
Retrain Compatible Models
Quick retrain to ensure feature compatibility

Author: AI Trading Systems
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.advanced_data_fetcher import AdvancedDataFetcher
from utils.feature_engineering import FeatureEngineer


def retrain_compatible_models():
    """Retrain models with current feature engineering for compatibility"""
    print("🔄 Retraining Compatible Models")
    print("=" * 50)
    
    # Initialize components
    data_fetcher = AdvancedDataFetcher()
    feature_engineer = FeatureEngineer()
    
    # Fetch data
    print("\n1. Fetching training data...")
    try:
        data = data_fetcher.fetch_historical_data('GC=F', '1y', '1h')
        if data is None or len(data) < 1000:
            print("   ⚠️  Using synthetic data...")
            data = generate_synthetic_data(5000)
        else:
            print(f"   ✅ Fetched {len(data)} real data points")
    except Exception as e:
        print(f"   ❌ Data fetch error: {e}")
        data = generate_synthetic_data(5000)
        
    # Create features
    print("\n2. Creating features...")
    try:
        features = feature_engineer.create_features_from_single_dataframe(data)
        
        # Create target
        future_returns = features['close'].pct_change().shift(-1)
        features['target'] = (future_returns > 0).astype(int)
        
        # Clean data
        features = features.dropna()
        
        print(f"   ✅ Features created: {features.shape[1]} columns, {len(features)} rows")
        
    except Exception as e:
        print(f"   ❌ Feature creation error: {e}")
        return False
        
    # Prepare training data
    print("\n3. Preparing training data...")
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in features.columns if col != 'target']
    X = features[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = features['target']
    
    # Remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)
        print(f"   🧹 Removed {len(constant_features)} constant features")
        
    print(f"   📊 Final feature count: {X.shape[1]}")
    print(f"   📊 Training samples: {len(X)}")
    print(f"   📊 Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n4. Training compatible models...")
    
    models = {}
    results = {}
    
    # Random Forest
    print("   🌲 Training Random Forest...")
    try:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        models['Random Forest'] = rf
        results['Random Forest'] = {'accuracy': rf_acc}
        print(f"      ✅ Random Forest: {rf_acc:.3f}")
        
    except Exception as e:
        print(f"      ❌ Random Forest error: {e}")
        
    # XGBoost
    print("   🚀 Training XGBoost...")
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        models['XGBoost'] = xgb_model
        results['XGBoost'] = {'accuracy': xgb_acc}
        print(f"      ✅ XGBoost: {xgb_acc:.3f}")
        
    except Exception as e:
        print(f"      ❌ XGBoost error: {e}")
        
    # LightGBM
    print("   💡 Training LightGBM...")
    try:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
            class_weight='balanced'
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        
        models['LightGBM'] = lgb_model
        results['LightGBM'] = {'accuracy': lgb_acc}
        print(f"      ✅ LightGBM: {lgb_acc:.3f}")
        
    except Exception as e:
        print(f"      ❌ LightGBM error: {e}")
        
    # Create ensemble
    if len(models) >= 2:
        print("   🧠 Creating ensemble...")
        try:
            predictions_list = []
            
            for name, model in models.items():
                if name == 'Random Forest':
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                predictions_list.append(pred)
                
            # Simple voting ensemble
            ensemble_pred = np.round(np.mean(predictions_list, axis=0)).astype(int)
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            
            results['Ensemble'] = {'accuracy': ensemble_acc}
            print(f"      ✅ Ensemble: {ensemble_acc:.3f}")
            
        except Exception as e:
            print(f"      ❌ Ensemble error: {e}")
            
    # Save models
    print("\n5. Saving compatible models...")
    
    os.makedirs('models/trained_models', exist_ok=True)
    
    saved_count = 0
    
    # Save models
    for name, model in models.items():
        try:
            filename = f"models/trained_models/{name.lower().replace(' ', '_')}_compatible.joblib"
            joblib.dump(model, filename)
            print(f"   ✅ {name} saved")
            saved_count += 1
        except Exception as e:
            print(f"   ❌ {name} save error: {e}")
            
    # Save scaler
    try:
        joblib.dump(scaler, 'models/trained_models/scaler_compatible.joblib')
        print(f"   ✅ Scaler saved")
    except Exception as e:
        print(f"   ❌ Scaler save error: {e}")
        
    # Save feature names
    try:
        feature_info = {
            'feature_names': list(X.columns),
            'feature_count': len(X.columns),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open('models/trained_models/feature_names_compatible.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"   ✅ Feature names saved")
    except Exception as e:
        print(f"   ❌ Feature names save error: {e}")
        
    # Save results
    try:
        results['metadata'] = {
            'training_timestamp': pd.Timestamp.now().isoformat(),
            'models_count': len(models),
            'feature_count': X.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        with open('models/trained_models/compatible_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   ✅ Results saved")
    except Exception as e:
        print(f"   ❌ Results save error: {e}")
        
    # Summary
    print("\n" + "=" * 50)
    print("🎉 COMPATIBLE MODEL TRAINING COMPLETE")
    print("=" * 50)
    
    if results:
        best_accuracy = max(result['accuracy'] for result in results.values() if 'accuracy' in result)
        print(f"📊 Best accuracy: {best_accuracy:.3f}")
        print(f"🤖 Models trained: {len(models)}")
        print(f"💾 Models saved: {saved_count}")
        print(f"📊 Features: {X.shape[1]}")
        
        print(f"\n📊 Model Performance:")
        for name, result in results.items():
            if 'accuracy' in result:
                accuracy = result['accuracy']
                status = "🎯" if accuracy >= 0.90 else "✅" if accuracy >= 0.85 else "👍"
                print(f"   {status} {name}: {accuracy:.3f}")
                
    print("=" * 50)
    
    return True


def generate_synthetic_data(num_samples=5000):
    """Generate synthetic training data"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=num_samples, freq='1H')
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


def main():
    """Main function"""
    success = retrain_compatible_models()
    
    if success:
        print("\n🚀 Compatible models ready!")
        print("🔄 Run: python test_trained_models.py")
    else:
        print("\n❌ Compatible training failed")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
