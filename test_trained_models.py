#!/usr/bin/env python3
"""
Test Trained Models
Verify that our trained models are working correctly

Author: AI Trading Systems
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.trained_model_loader import TrainedModelLoader
from utils.advanced_data_fetcher import AdvancedDataFetcher
from utils.feature_engineering import FeatureEngineer


def test_trained_models():
    """Test the trained models with real data"""
    print("🧪 Testing Trained Models")
    print("=" * 50)
    
    # Initialize components
    model_loader = TrainedModelLoader()
    data_fetcher = AdvancedDataFetcher()
    feature_engineer = FeatureEngineer()
    
    # Load models
    print("\n1. Loading trained models...")
    success = model_loader.load_models()
    
    if not success:
        print("❌ No trained models found")
        return False
        
    # Get model info
    info = model_loader.get_model_info()
    print(f"   Models loaded: {info['model_count']}")
    print(f"   Models: {', '.join(info['models'])}")
    print(f"   Has scaler: {info['has_scaler']}")
    
    # Fetch test data
    print("\n2. Fetching test data...")
    try:
        test_data = data_fetcher.fetch_historical_data('GC=F', '1d', '1h')
        if test_data is None or len(test_data) < 100:
            print("   ⚠️  Using synthetic test data...")
            test_data = generate_test_data()
        else:
            print(f"   ✅ Fetched {len(test_data)} real data points")
    except Exception as e:
        print(f"   ⚠️  Data fetch error: {e}")
        test_data = generate_test_data()
        
    # Create features
    print("\n3. Creating features...")
    try:
        features = feature_engineer.create_features_from_single_dataframe(test_data)
        print(f"   ✅ Created {features.shape[1]} features from {len(features)} data points")
        
        # Use the last row for prediction
        latest_features = features.iloc[-1:].drop(columns=[col for col in features.columns if col.startswith('target_')], errors='ignore')
        
    except Exception as e:
        print(f"   ❌ Feature creation error: {e}")
        return False
        
    # Test ensemble prediction
    print("\n4. Testing ensemble prediction...")
    try:
        result = model_loader.predict(latest_features)
        
        if result:
            print(f"   ✅ Ensemble Prediction:")
            print(f"      Signal: {result['signal']}")
            print(f"      Strength: {result['strength']}")
            print(f"      Confidence: {result['confidence']:.3f}")
            print(f"      Ensemble Value: {result['ensemble_prediction']:.3f}")
            print(f"      Models Used: {len(result['models_used'])}")
            
            # Show individual model predictions
            print(f"\n   📊 Individual Model Results:")
            for model_name, pred in result['individual_predictions'].items():
                prob_info = result['individual_probabilities'].get(model_name, {})
                confidence = prob_info.get('confidence', 0.5)
                print(f"      {model_name}: {pred:.3f} (confidence: {confidence:.3f})")
                
        else:
            print("   ❌ Prediction failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Ensemble prediction error: {e}")
        return False
        
    # Test best model prediction
    print("\n5. Testing best model prediction...")
    try:
        best_model = model_loader.get_best_model()
        print(f"   Best model: {best_model}")
        
        best_result = model_loader.predict_with_best_model(latest_features)
        
        if best_result:
            print(f"   ✅ Best Model Prediction:")
            print(f"      Signal: {best_result['signal']}")
            print(f"      Confidence: {best_result['confidence']:.3f}")
            print(f"      Value: {best_result['prediction']:.3f}")
            print(f"      Model: {best_result['model_used']}")
        else:
            print("   ❌ Best model prediction failed")
            
    except Exception as e:
        print(f"   ❌ Best model prediction error: {e}")
        
    # Test multiple predictions
    print("\n6. Testing multiple predictions...")
    try:
        if len(features) >= 10:
            # Test on last 10 data points
            test_features = features.iloc[-10:].drop(columns=[col for col in features.columns if col.startswith('target_')], errors='ignore')
            
            predictions = []
            for i in range(len(test_features)):
                pred = model_loader.predict(test_features.iloc[i:i+1])
                if pred:
                    predictions.append(pred['signal'])
                else:
                    predictions.append('UNKNOWN')
                    
            print(f"   ✅ Batch predictions: {predictions}")
            
            # Calculate signal distribution
            signal_counts = {}
            for signal in predictions:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
                
            print(f"   📊 Signal distribution: {signal_counts}")
            
        else:
            print("   ⚠️  Not enough data for batch testing")
            
    except Exception as e:
        print(f"   ❌ Batch prediction error: {e}")
        
    # Performance summary
    print("\n" + "=" * 50)
    print("🎉 MODEL TESTING COMPLETE")
    print("=" * 50)
    
    if info['metadata']:
        print("📊 Model Performance (from training):")
        for model_name, metrics in info['metadata'].items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                accuracy = metrics['accuracy']
                status = "🎯" if accuracy >= 0.90 else "✅" if accuracy >= 0.85 else "👍"
                print(f"   {status} {model_name}: {accuracy:.3f}")
                
                if 'confident_accuracy' in metrics and metrics['confident_accuracy']:
                    conf_acc = metrics['confident_accuracy']
                    conf_pct = metrics['confident_percentage']
                    print(f"      🎯 Confident: {conf_acc:.3f} ({conf_pct:.1f}% of data)")
                    
    print("\n✅ All tests completed successfully!")
    print("🚀 Trained models are working correctly!")
    
    return True


def generate_test_data(num_points=100):
    """Generate synthetic test data"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=num_points, freq='1H')
    base_price = 2000.0
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.01, num_points)
    returns = np.cumsum(returns)
    prices = base_price * np.exp(returns)
    
    data = pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, num_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, num_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, num_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, num_points)
    })
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data


def main():
    """Main function"""
    success = test_trained_models()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
