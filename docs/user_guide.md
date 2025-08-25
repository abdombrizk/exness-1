# Gold Trading AI - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [System Overview](#system-overview)
3. [Training Models](#training-models)
4. [Using the GUI Application](#using-the-gui-application)
5. [Database Management](#database-management)
6. [Performance Monitoring](#performance-monitoring)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for real-time data

### Quick Installation
```bash
# Clone repository
git clone https://github.com/your-repo/gold-trading-ai.git
cd gold-trading-ai

# Install dependencies
pip install -r requirements.txt

# Train models (first time only)
python ml_system/advanced_trainer.py

# Launch application
python gui_app/main_application.py
```

## System Overview

### Architecture
The Gold Trading AI system consists of four main components:

1. **ML System** (`ml_system/`): Advanced machine learning models
2. **GUI Application** (`gui_app/`): Professional desktop interface
3. **Database** (`database/`): Comprehensive data storage
4. **Testing** (`tests/`): Validation and testing suite

### Key Features
- **>90% Accuracy**: Ensemble ML models with hyperparameter optimization
- **Real-time Data**: Live gold price feeds and predictions
- **Professional Interface**: Bloomberg Terminal-style GUI
- **Comprehensive Database**: Performance tracking and historical data

## Training Models

### Automatic Training
The system includes an advanced training pipeline that automatically:
- Fetches real gold price data
- Engineers 129+ features
- Trains ensemble models (Random Forest, XGBoost, LightGBM)
- Optimizes hyperparameters
- Validates performance

```bash
python ml_system/advanced_trainer.py
```

### Training Process
1. **Data Collection**: Fetches historical gold prices from multiple sources
2. **Feature Engineering**: Creates technical indicators, price patterns, and statistical features
3. **Target Selection**: Automatically selects the best prediction target
4. **Model Training**: Trains multiple models with hyperparameter optimization
5. **Ensemble Creation**: Combines models using weighted voting
6. **Validation**: Uses time series cross-validation for realistic performance

### Expected Results
- **Training Time**: 10-30 minutes depending on hardware
- **Target Accuracy**: >90% on validation data
- **Models Created**: Random Forest, XGBoost, LightGBM, Ensemble
- **Features**: 129+ comprehensive features

## Using the GUI Application

### Launching the Application
```bash
python gui_app/main_application.py
```

### Interface Layout
The application features a professional split-screen layout:

**Left Panel (50%): Data Input & Controls**
- **Market Data**: Current price, OHLCV data, real-time updates
- **Manual Input**: Custom data entry for testing
- **Controls**: Real-time toggle, refresh, model loading
- **Model Information**: Loaded model details and status

**Right Panel (50%): Results & Analysis**
- **AI Predictions**: Signal, confidence, probability
- **Price Chart**: Real-time candlestick chart with indicators
- **Risk Management**: Position sizing, stop-loss, take-profit

### Key Features

#### Real-time Data
- **Automatic Updates**: 30-second intervals (configurable)
- **Live Indicator**: Shows connection status
- **Manual Refresh**: Force data update
- **Fallback Data**: Synthetic data when real data unavailable

#### AI Predictions
- **Signal Types**: BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
- **Confidence Score**: Model confidence (0-100%)
- **Probability**: Prediction probability
- **Update Frequency**: Real-time with data updates

#### Manual Data Input
For testing and analysis:
1. Enter custom price data
2. Set volume and OHLC values
3. Click "Apply Manual Data"
4. View AI predictions on custom data

#### Risk Management
Automatic calculations:
- **Position Size**: Based on risk tolerance
- **Stop Loss**: 2% below entry (configurable)
- **Take Profit**: 4% above entry (configurable)
- **Risk/Reward Ratio**: Calculated automatically

### Settings and Configuration
Access settings through the "Settings" button:
- **Update Interval**: Change real-time update frequency
- **Data Sources**: Configure data providers
- **Model Parameters**: Adjust prediction thresholds

## Database Management

### Database Structure
The system uses SQLite with the following tables:
- **market_data**: Historical price data
- **predictions**: AI predictions and outcomes
- **model_performance**: Model accuracy metrics
- **trading_signals**: Trading recommendations
- **risk_management**: Risk calculations
- **backtesting_results**: Backtest performance

### Accessing Database
```python
from database.trading_database import DatabaseManager

# Initialize database
db_manager = DatabaseManager()

# Get market data
market_data = db_manager.db.get_market_data()

# Get predictions
predictions = db_manager.db.get_predictions()

# Generate performance report
report = db_manager.generate_performance_report()
```

### Data Export
Export data for analysis:
```python
# Export predictions to CSV
db_manager.db.export_data('predictions', 'predictions.csv')

# Export market data
db_manager.db.export_data('market_data', 'market_data.csv')
```

### Database Maintenance
- **Cleanup**: Automatically removes old data (365 days)
- **Backup**: Create database backups
- **Validation**: Check data integrity
- **Statistics**: Monitor database size and performance

## Performance Monitoring

### Model Performance Tracking
The system continuously monitors:
- **Prediction Accuracy**: Real-time accuracy calculation
- **Confidence Scores**: Model confidence distribution
- **Signal Distribution**: BUY/SELL/HOLD ratios
- **Performance Trends**: Accuracy over time

### Accessing Performance Data
```python
# Get accuracy for specific model
accuracy = db_manager.db.calculate_prediction_accuracy('LightGBM', days_back=30)

# Get performance summary
summary = db_manager.db.get_performance_summary()

# Generate comprehensive report
report = db_manager.generate_performance_report()
```

### Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confidence**: Average model confidence
- **Win Rate**: Percentage of profitable signals

## Troubleshooting

### Common Issues

#### Model Loading Error
**Problem**: "Model: Not Found" in GUI
**Solution**:
```bash
# Retrain models
python ml_system/advanced_trainer.py

# Or load specific model
# Use "Load Model" button in GUI
```

#### Database Connection Error
**Problem**: Database operations fail
**Solution**:
```bash
# Reinitialize database
python database/trading_database.py

# Check database file permissions
# Ensure sufficient disk space
```

#### Real-time Data Error
**Problem**: No real-time data updates
**Solution**:
- Check internet connection
- Verify data source availability
- Use manual data input as fallback
- Check firewall settings

#### GUI Launch Error
**Problem**: Application won't start
**Solution**:
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.8+

# Run with error output
python gui_app/main_application.py
```

#### Low Prediction Accuracy
**Problem**: Model accuracy below expectations
**Solution**:
- Retrain with more recent data
- Check data quality
- Verify feature engineering
- Consider ensemble methods

### Performance Optimization

#### Memory Usage
- **Large Datasets**: Increase available RAM
- **Feature Engineering**: Monitor memory during training
- **Database**: Regular cleanup of old data

#### Processing Speed
- **CPU**: Use multi-core processing for training
- **Storage**: SSD recommended for database operations
- **Network**: Stable internet for real-time data

#### Accuracy Improvement
- **More Data**: Increase training dataset size
- **Feature Engineering**: Add domain-specific features
- **Hyperparameters**: Fine-tune model parameters
- **Ensemble Methods**: Combine multiple models

### Getting Help

#### Log Files
Check log files for detailed error information:
- Application logs in `logs/` directory
- Database logs in database error messages
- Model training logs in console output

#### Debug Mode
Run components in debug mode:
```bash
# Debug model training
python -m pdb ml_system/advanced_trainer.py

# Debug GUI application
python -m pdb gui_app/main_application.py
```

#### Support Resources
- **Documentation**: Check `docs/` directory
- **Test Suite**: Run `python tests/comprehensive_tests.py`
- **GitHub Issues**: Report bugs and feature requests
- **Community**: Join discussions and get help

### Best Practices

#### Data Management
- Regular database backups
- Monitor data quality
- Clean old data periodically
- Validate data integrity

#### Model Management
- Retrain models regularly
- Monitor performance metrics
- Keep multiple model versions
- Document model changes

#### System Maintenance
- Update dependencies regularly
- Monitor system resources
- Run test suite before updates
- Keep configuration backups

---

For more detailed information, see:
- [API Reference](api_reference.md)
- [Installation Guide](installation.md)
- [Developer Guide](developer_guide.md)
