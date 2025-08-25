#!/usr/bin/env python3
"""
Gold Trading AI - System Demonstration
======================================

Demonstrates the complete Gold Trading AI system functionality
including ML training, prediction, and professional interface.

Author: AI Trading Systems
Version: 2.0.0
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def print_banner():
    """Print professional system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🥇 GOLD TRADING AI - SYSTEM DEMONSTRATION                 ║
║                                                                              ║
║                    🎯 Target: >90% Accuracy ML Predictions                   ║
║                    🖥️  Bloomberg Terminal-Style Interface                     ║
║                    🗄️  Comprehensive Database Integration                     ║
║                    🧪 Full Testing & Validation Suite                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_requirements():
    """Check system requirements and dependencies"""
    print("🔍 Checking System Requirements...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version compatible")
    
    # Check core dependencies
    required_packages = ['numpy', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing core dependencies...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    print("✅ All system requirements met")
    return True

def generate_synthetic_data(size=1000):
    """Generate synthetic gold price data for demonstration"""
    print("\n📊 Generating Synthetic Market Data...")
    
    # Generate realistic gold price data
    np.random.seed(42)  # For reproducible results
    
    # Start with base price
    base_price = 2000.0
    dates = pd.date_range(start='2023-01-01', periods=size, freq='1H')
    
    # Generate price movements with realistic volatility
    returns = np.random.normal(0, 0.01, size)  # 1% hourly volatility
    prices = [base_price]
    
    for i in range(1, size):
        # Add some trend and mean reversion
        trend = 0.0001 * np.sin(i / 100)  # Slight cyclical trend
        mean_reversion = -0.001 * (prices[-1] - base_price) / base_price
        
        price_change = returns[i] + trend + mean_reversion
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Create OHLC data
    data = []
    for i in range(size):
        open_price = prices[i]
        close_price = prices[i] if i == size - 1 else prices[i + 1]
        
        # Generate high and low based on volatility
        volatility = abs(returns[i]) * 2
        high_price = max(open_price, close_price) * (1 + volatility)
        low_price = min(open_price, close_price) * (1 - volatility)
        
        volume = int(np.random.lognormal(8, 1))  # Realistic volume distribution
        
        data.append({
            'timestamp': dates[i],
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} data points")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"   Average volume: {df['volume'].mean():,.0f}")
    
    return df

def create_technical_features(data):
    """Create technical analysis features"""
    print("\n🔧 Engineering Technical Features...")
    
    df = data.copy()
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['price_change'].rolling(20).std()
    df['high_low_pct'] = (df['high'] - df['low']) / df['close']
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    feature_count = len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])
    print(f"✅ Created {feature_count} technical features")
    
    return df

def create_targets(data):
    """Create prediction targets"""
    print("\n🎯 Creating Prediction Targets...")
    
    df = data.copy()
    
    # Direction target (next hour price direction)
    df['target_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Strong movement target (>1% move)
    df['next_return'] = df['close'].pct_change().shift(-1)
    df['target_strong_move'] = (abs(df['next_return']) > 0.01).astype(int)
    
    # Trend target (next 4 hours trend)
    df['future_price'] = df['close'].shift(-4)
    df['target_trend'] = ((df['future_price'] - df['close']) / df['close'] > 0.005).astype(int)
    
    print("✅ Created prediction targets:")
    print(f"   Direction: {df['target_direction'].value_counts().to_dict()}")
    print(f"   Strong moves: {df['target_strong_move'].value_counts().to_dict()}")
    print(f"   Trends: {df['target_trend'].value_counts().to_dict()}")
    
    return df

def train_simple_model(data):
    """Train a simple ML model for demonstration"""
    print("\n🤖 Training Machine Learning Model...")
    
    # Prepare features
    feature_cols = [col for col in data.columns if col.startswith(('sma_', 'ema_', 'rsi', 'macd', 'bb_', 'price_', 'volatility', 'volume_'))]
    
    # Remove rows with NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 100:
        print("❌ Insufficient data for training")
        return None, None, 0.0
    
    X = clean_data[feature_cols]
    y = clean_data['target_direction']
    
    # Simple train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(feature_cols)}")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully")
        print(f"   Accuracy: {accuracy:.1%}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   Top features: {', '.join(feature_importance.head(3)['feature'].tolist())}")
        
        return model, scaler, accuracy
        
    except ImportError:
        print("⚠️  scikit-learn not available, using simple baseline model")
        
        # Simple baseline: predict based on recent trend
        recent_trend = (clean_data['close'].iloc[-10:].iloc[-1] > clean_data['close'].iloc[-10:].iloc[0])
        baseline_pred = [1 if recent_trend else 0] * len(y_test)
        accuracy = accuracy_score(y_test, baseline_pred)
        
        print(f"✅ Baseline model created")
        print(f"   Accuracy: {accuracy:.1%}")
        
        return "baseline", None, accuracy

def make_predictions(model, scaler, data, feature_cols):
    """Make real-time predictions"""
    print("\n🔮 Making Real-time Predictions...")
    
    if model == "baseline":
        # Simple baseline prediction
        recent_trend = data['close'].iloc[-5:].pct_change().mean()
        signal = "BUY" if recent_trend > 0 else "SELL"
        confidence = min(abs(recent_trend) * 100, 0.8)
        
        prediction = {
            'signal': signal,
            'confidence': confidence,
            'probability': 0.5 + (recent_trend * 10),
            'model': 'Baseline Trend'
        }
    else:
        try:
            # Get latest data point
            latest_data = data.iloc[-1:][feature_cols].fillna(0)
            
            if scaler:
                latest_scaled = scaler.transform(latest_data)
            else:
                latest_scaled = latest_data.values
            
            # Make prediction
            prob = model.predict_proba(latest_scaled)[0]
            pred = model.predict(latest_scaled)[0]
            
            signal = "BUY" if pred == 1 else "SELL"
            confidence = max(prob)
            
            prediction = {
                'signal': signal,
                'confidence': confidence,
                'probability': prob[1],
                'model': 'Random Forest'
            }
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            prediction = {
                'signal': 'HOLD',
                'confidence': 0.5,
                'probability': 0.5,
                'model': 'Error Fallback'
            }
    
    print(f"✅ Prediction generated:")
    print(f"   Signal: {prediction['signal']}")
    print(f"   Confidence: {prediction['confidence']:.1%}")
    print(f"   Model: {prediction['model']}")
    
    return prediction

def display_market_summary(data):
    """Display current market summary"""
    print("\n📈 Current Market Summary")
    print("=" * 30)
    
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest
    
    price_change = latest['close'] - previous['close']
    price_change_pct = (price_change / previous['close']) * 100
    
    print(f"Current Price: ${latest['close']:.2f}")
    print(f"Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")
    print(f"High: ${latest['high']:.2f}")
    print(f"Low: ${latest['low']:.2f}")
    print(f"Volume: {latest['volume']:,}")
    
    # Technical indicators
    if 'rsi' in data.columns and not pd.isna(latest['rsi']):
        print(f"RSI: {latest['rsi']:.1f}")
    
    if 'sma_20' in data.columns and not pd.isna(latest['sma_20']):
        print(f"SMA(20): ${latest['sma_20']:.2f}")

def simulate_risk_management(prediction, current_price):
    """Simulate risk management calculations"""
    print("\n⚖️  Risk Management Analysis")
    print("=" * 30)
    
    position_size = 1.0  # Standard lot
    
    if prediction['signal'] == 'BUY':
        entry_price = current_price
        stop_loss = entry_price * 0.98  # 2% stop loss
        take_profit = entry_price * 1.04  # 4% take profit
    elif prediction['signal'] == 'SELL':
        entry_price = current_price
        stop_loss = entry_price * 1.02  # 2% stop loss
        take_profit = entry_price * 0.96  # 4% take profit
    else:
        print("No position recommended")
        return
    
    risk_amount = abs(entry_price - stop_loss) * position_size
    reward_amount = abs(take_profit - entry_price) * position_size
    risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
    
    print(f"Signal: {prediction['signal']}")
    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Stop Loss: ${stop_loss:.2f}")
    print(f"Take Profit: ${take_profit:.2f}")
    print(f"Position Size: {position_size} lots")
    print(f"Risk Amount: ${risk_amount:.2f}")
    print(f"Reward Amount: ${reward_amount:.2f}")
    print(f"Risk/Reward: 1:{risk_reward_ratio:.1f}")

def run_system_demonstration():
    """Run complete system demonstration"""
    print_banner()
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please install required dependencies.")
        return False
    
    try:
        # Generate data
        market_data = generate_synthetic_data(500)
        
        # Create features
        featured_data = create_technical_features(market_data)
        
        # Create targets
        target_data = create_targets(featured_data)
        
        # Train model
        feature_cols = [col for col in target_data.columns if col.startswith(('sma_', 'ema_', 'rsi', 'macd', 'bb_', 'price_', 'volatility', 'volume_'))]
        model, scaler, accuracy = train_simple_model(target_data)
        
        if model is None:
            print("❌ Model training failed")
            return False
        
        # Make predictions
        prediction = make_predictions(model, scaler, target_data, feature_cols)
        
        # Display results
        display_market_summary(market_data)
        
        # Risk management
        current_price = market_data.iloc[-1]['close']
        simulate_risk_management(prediction, current_price)
        
        # System summary
        print("\n" + "=" * 60)
        print("🎉 GOLD TRADING AI SYSTEM DEMONSTRATION COMPLETE")
        print("=" * 60)
        print(f"✅ Data Processing: {len(market_data)} market data points")
        print(f"✅ Feature Engineering: {len(feature_cols)} technical features")
        print(f"✅ Model Training: {accuracy:.1%} accuracy achieved")
        print(f"✅ Real-time Prediction: {prediction['signal']} signal generated")
        print(f"✅ Risk Management: Position sizing and risk calculations")
        
        print("\n🚀 System Features Demonstrated:")
        print("   📊 Synthetic market data generation")
        print("   🔧 Advanced feature engineering")
        print("   🤖 Machine learning model training")
        print("   🔮 Real-time prediction generation")
        print("   ⚖️  Professional risk management")
        print("   📈 Market analysis and reporting")
        
        print("\n💡 Next Steps:")
        print("   1. Install full dependencies: pip install -r requirements.txt")
        print("   2. Run complete system: python launch.py")
        print("   3. Launch GUI application: python -m src.gui.application")
        print("   4. Train production models: python -m src.core.models.trainer")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_system_demonstration()
    sys.exit(0 if success else 1)
