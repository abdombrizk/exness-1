# Data Directory

This directory contains all data files for the Gold Trading AI system, organized by type and processing stage.

## Directory Structure

```
data/
├── raw/                        # Raw, unprocessed data
│   ├── market_data/           # Raw market data files
│   │   ├── gold_prices_2023.csv
│   │   ├── gold_prices_2024.csv
│   │   └── economic_indicators.csv
│   ├── news/                  # News and sentiment data
│   └── external/              # External data sources
├── processed/                 # Processed and cleaned data
│   ├── features/              # Engineered features
│   │   ├── technical_indicators.csv
│   │   ├── price_patterns.csv
│   │   └── market_regime.csv
│   ├── targets/               # Target variables
│   │   ├── direction_targets.csv
│   │   └── volatility_targets.csv
│   └── datasets/              # Final training datasets
│       ├── training_set.csv
│       ├── validation_set.csv
│       └── test_set.csv
├── models/                    # Trained ML models
│   ├── random_forest_20240115_143022/
│   │   ├── model.joblib
│   │   ├── scaler.joblib
│   │   └── metadata.json
│   ├── xgboost_20240115_143045/
│   └── ensemble_20240115_143100/
├── database/                  # Database files
│   ├── trading_data.db       # Main SQLite database
│   ├── trading_data.db-wal   # Write-ahead log
│   └── trading_data.db-shm   # Shared memory
├── backups/                   # Data backups
│   ├── daily/                # Daily backups
│   ├── weekly/               # Weekly backups
│   └── monthly/              # Monthly backups
├── cache/                     # Cached data for performance
│   ├── features/             # Cached feature calculations
│   └── predictions/          # Cached predictions
└── exports/                   # Exported data and reports
    ├── predictions.csv       # Prediction exports
    ├── performance_report.pdf
    └── trading_signals.csv
```

## Data Types

### 📊 Raw Data (`data/raw/`)

**Market Data**
- Gold futures prices (GC=F)
- Gold spot prices (XAUUSD=X)
- Gold ETF prices (GLD)
- Volume and open interest data
- Intraday tick data

**Economic Indicators**
- US Dollar Index (DXY)
- 10-Year Treasury yields (TNX)
- VIX volatility index
- Currency exchange rates
- Inflation data (CPI, PPI)

**News and Sentiment**
- Financial news articles
- Social media sentiment
- Economic calendar events
- Central bank announcements

### 🔧 Processed Data (`data/processed/`)

**Technical Features**
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD)
- Volatility measures (ATR, Bollinger Bands)
- Volume indicators (OBV, VWAP)
- Price patterns and formations

**Fundamental Features**
- Economic indicator correlations
- Currency strength measures
- Interest rate differentials
- Inflation-adjusted prices

**Target Variables**
- Price direction (up/down)
- Volatility breakouts
- Trend changes
- Strong movements (>2% moves)

### 🤖 Models (`data/models/`)

Each model directory contains:
- `model.joblib` - Trained model object
- `scaler.joblib` - Feature scaler
- `metadata.json` - Model metadata and performance metrics

**Model Naming Convention:**
`{model_type}_{YYYYMMDD}_{HHMMSS}/`

Example: `lightgbm_20240115_143022/`

### 🗄️ Database (`data/database/`)

**SQLite Database Structure:**
- `market_data` - Historical price data
- `predictions` - Model predictions and outcomes
- `model_performance` - Model accuracy metrics
- `trading_signals` - Trading recommendations
- `risk_management` - Risk calculations
- `backtesting_results` - Backtest performance

## Data Management

### Data Collection

**Automated Collection:**
```python
from src.core.data import DataCollector

collector = DataCollector()
collector.fetch_gold_data(period="1y", interval="1h")
collector.fetch_economic_data()
collector.save_to_raw()
```

**Manual Data Import:**
```python
import pandas as pd

# Import CSV data
data = pd.read_csv('data/raw/market_data/custom_data.csv')
collector.import_data(data, source="custom")
```

### Data Processing

**Feature Engineering:**
```python
from src.core.models.feature_engineering import AdvancedFeatureEngineer

engineer = AdvancedFeatureEngineer()
features = engineer.create_comprehensive_features(raw_data)
features.to_csv('data/processed/features/technical_indicators.csv')
```

**Target Creation:**
```python
from src.core.models.trainer import AdvancedMLTrainer

trainer = AdvancedMLTrainer()
data_with_targets = trainer.create_targets(processed_data)
```

### Data Validation

**Quality Checks:**
```python
from src.utils.validators import DataValidator

validator = DataValidator()
validation_report = validator.validate_market_data('data/raw/market_data/')

if validation_report['is_valid']:
    print("Data validation passed")
else:
    print(f"Validation errors: {validation_report['errors']}")
```

**Data Integrity:**
```python
# Check for missing values
missing_data = data.isnull().sum()

# Check for outliers
outliers = validator.detect_outliers(data['close'])

# Validate price consistency (OHLC)
price_consistency = validator.validate_ohlc_consistency(data)
```

## Data Backup and Recovery

### Automated Backups

**Daily Backups:**
```bash
# Automated via cron job
0 2 * * * python scripts/backup_data.py --type daily
```

**Database Backups:**
```python
from src.core.database import DatabaseManager

db_manager = DatabaseManager()
backup_path = db_manager.backup_database('data/backups/daily/')
```

### Data Recovery

**Restore from Backup:**
```python
# Restore database
db_manager.restore_database('data/backups/daily/trading_data_20240115.db')

# Restore model
model_manager.restore_model('data/backups/models/best_model_20240115.tar.gz')
```

## Data Security and Privacy

### Data Encryption

**Sensitive Data:**
```python
from src.utils.security import DataEncryption

encryptor = DataEncryption()

# Encrypt sensitive files
encryptor.encrypt_file('data/raw/api_responses.json')

# Decrypt when needed
decrypted_data = encryptor.decrypt_file('data/raw/api_responses.json.enc')
```

### Access Control

**File Permissions:**
```bash
# Set appropriate permissions
chmod 600 data/database/trading_data.db  # Database files
chmod 700 data/backups/                  # Backup directory
chmod 644 data/processed/                # Processed data (read-only)
```

## Data Monitoring

### Data Quality Monitoring

**Automated Checks:**
```python
from src.utils.monitoring import DataMonitor

monitor = DataMonitor()

# Check data freshness
freshness_report = monitor.check_data_freshness('data/raw/market_data/')

# Monitor data volume
volume_report = monitor.check_data_volume()

# Detect anomalies
anomalies = monitor.detect_data_anomalies('data/processed/features/')
```

### Performance Monitoring

**Storage Usage:**
```python
import os

def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size

# Monitor storage usage
raw_size = get_directory_size('data/raw/')
processed_size = get_directory_size('data/processed/')
models_size = get_directory_size('data/models/')
```

## Data Lifecycle Management

### Data Retention Policy

**Retention Rules:**
- Raw data: Keep for 2 years
- Processed data: Keep for 1 year
- Model data: Keep best 5 models per type
- Database: Archive after 1 year
- Backups: Keep daily for 30 days, weekly for 12 weeks, monthly for 12 months

**Automated Cleanup:**
```python
from src.utils.cleanup import DataCleanup

cleanup = DataCleanup()

# Clean old raw data
cleanup.clean_old_data('data/raw/', days=730)  # 2 years

# Clean old processed data
cleanup.clean_old_data('data/processed/', days=365)  # 1 year

# Clean old models
cleanup.clean_old_models('data/models/', keep_count=5)
```

### Data Archival

**Archive Old Data:**
```python
import tarfile
from datetime import datetime, timedelta

def archive_old_data(source_dir, archive_dir, days_old=365):
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    # Create archive
    archive_name = f"archive_{datetime.now().strftime('%Y%m%d')}.tar.gz"
    archive_path = os.path.join(archive_dir, archive_name)
    
    with tarfile.open(archive_path, 'w:gz') as tar:
        for file_path in old_files:
            tar.add(file_path)
```

## Best Practices

### 1. Data Organization
- Keep raw data immutable
- Use consistent naming conventions
- Separate data by processing stage
- Document data sources and transformations

### 2. Data Quality
- Validate all incoming data
- Monitor data quality metrics
- Handle missing values appropriately
- Detect and handle outliers

### 3. Performance
- Use efficient file formats (Parquet, HDF5)
- Implement data caching for frequently accessed data
- Compress large datasets
- Use database indexing for fast queries

### 4. Security
- Encrypt sensitive data
- Use appropriate file permissions
- Implement access logging
- Regular security audits

### 5. Backup and Recovery
- Automated daily backups
- Test recovery procedures regularly
- Store backups in multiple locations
- Document recovery procedures

This data management system ensures reliable, secure, and efficient handling of all data in the Gold Trading AI system.
