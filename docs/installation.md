# Gold Trading AI - Installation Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation](#detailed-installation)
4. [Dependency Management](#dependency-management)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10, macOS 10.15, or Ubuntu 18.04
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM
- **Storage**: 2GB free space
- **Internet**: Required for real-time data and package installation

### Recommended Requirements
- **Operating System**: Windows 11, macOS 12, or Ubuntu 20.04
- **Python**: 3.9 or 3.10
- **Memory**: 8GB RAM or higher
- **Storage**: 5GB free space (SSD recommended)
- **Internet**: Stable broadband connection

### Hardware Recommendations
- **CPU**: Multi-core processor (4+ cores recommended for training)
- **GPU**: Optional, but beneficial for deep learning models
- **Storage**: SSD for better database performance
- **Network**: Stable internet for real-time data feeds

## Quick Installation

### One-Command Setup
```bash
# Clone and setup in one go
git clone https://github.com/your-repo/gold-trading-ai.git && \
cd gold-trading-ai && \
pip install -r requirements.txt && \
python ml_system/advanced_trainer.py && \
python gui_app/main_application.py
```

### Step-by-Step Quick Setup
```bash
# 1. Clone repository
git clone https://github.com/your-repo/gold-trading-ai.git
cd gold-trading-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (first time only)
python ml_system/advanced_trainer.py

# 4. Launch application
python gui_app/main_application.py
```

## Detailed Installation

### Step 1: Python Installation

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer with "Add Python to PATH" checked
3. Verify installation:
```cmd
python --version
pip --version
```

#### macOS
```bash
# Using Homebrew (recommended)
brew install python

# Or download from python.org
# Verify installation
python3 --version
pip3 --version
```

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

### Step 2: Virtual Environment (Recommended)

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv gold_trading_ai_env

# Activate virtual environment
# Windows:
gold_trading_ai_env\Scripts\activate

# macOS/Linux:
source gold_trading_ai_env/bin/activate

# Verify activation (should show environment name)
which python
```

#### Deactivate Virtual Environment
```bash
deactivate
```

### Step 3: Repository Setup

#### Clone Repository
```bash
# Using HTTPS
git clone https://github.com/your-repo/gold-trading-ai.git

# Using SSH (if configured)
git clone git@github.com:your-repo/gold-trading-ai.git

# Navigate to directory
cd gold-trading-ai
```

#### Verify Repository Structure
```bash
# Check directory structure
ls -la

# Should see:
# - ml_system/
# - gui_app/
# - database/
# - tests/
# - docs/
# - requirements.txt
# - README.md
```

### Step 4: Dependency Installation

#### Install Core Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

#### Manual Dependency Installation
If requirements.txt fails, install manually:
```bash
# Core ML libraries
pip install numpy pandas scikit-learn

# Advanced ML libraries
pip install xgboost lightgbm optuna

# Technical analysis
pip install TA-Lib pandas-ta

# Data sources
pip install yfinance

# GUI libraries
pip install matplotlib tkinter

# Database
pip install sqlite3

# Testing
pip install pytest unittest
```

### Step 5: TA-Lib Installation

TA-Lib requires special installation:

#### Windows
```bash
# Download TA-Lib wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

# Install downloaded wheel
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl

# Or use conda
conda install -c conda-forge ta-lib
```

#### macOS
```bash
# Install dependencies
brew install ta-lib

# Install Python package
pip install TA-Lib
```

#### Linux
```bash
# Install dependencies
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Install Python package
pip install TA-Lib
```

## Dependency Management

### Requirements Files

#### requirements.txt (Core Dependencies)
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
optuna>=3.0.0
TA-Lib>=0.4.24
pandas-ta>=0.3.14b
yfinance>=0.1.87
matplotlib>=3.5.0
```

#### requirements-dev.txt (Development Dependencies)
```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.991
jupyter>=1.0.0
```

### Dependency Updates
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade numpy

# Check outdated packages
pip list --outdated
```

## Configuration

### Environment Variables
Create `.env` file in project root:
```bash
# Data source configuration
GOLD_DATA_SOURCE=yfinance
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Database configuration
DATABASE_PATH=database/trading_data.db
DATABASE_BACKUP_PATH=backups/

# Model configuration
MODEL_PATH=ml_system/models/
MODEL_CACHE_SIZE=1000

# GUI configuration
GUI_THEME=professional
UPDATE_INTERVAL=30

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/application.log
```

### Configuration Files

#### config/settings.json
```json
{
  "data_sources": {
    "primary": "yfinance",
    "fallback": "synthetic",
    "update_interval": 30
  },
  "models": {
    "target_accuracy": 0.90,
    "ensemble_weights": "auto",
    "confidence_threshold": 0.6
  },
  "gui": {
    "theme": "professional",
    "auto_refresh": true,
    "chart_timeframe": "1h"
  },
  "database": {
    "cleanup_days": 365,
    "backup_frequency": "daily",
    "validation_enabled": true
  }
}
```

### Directory Structure Setup
```bash
# Create necessary directories
mkdir -p logs
mkdir -p backups
mkdir -p config
mkdir -p ml_system/models
mkdir -p tests/reports
mkdir -p docs/assets

# Set permissions (Linux/macOS)
chmod 755 logs backups config
chmod 644 config/settings.json
```

## Verification

### Installation Verification
```bash
# Run verification script
python -c "
import numpy as np
import pandas as pd
import sklearn
import xgboost
import lightgbm
import talib
import yfinance
import matplotlib
print('All core dependencies installed successfully!')
"
```

### Component Testing
```bash
# Test ML system
python -c "from ml_system.advanced_trainer import AdvancedMLTrainer; print('ML system OK')"

# Test database
python -c "from database.trading_database import TradingDatabase; print('Database OK')"

# Test GUI components
python -c "from gui_app.main_application import DataManager; print('GUI components OK')"
```

### Full System Test
```bash
# Run comprehensive test suite
python tests/comprehensive_tests.py

# Expected output:
# 🧪 Starting Comprehensive Test Suite
# ✅ All tests passed
# 🎉 System ready for production!
```

### Performance Benchmark
```bash
# Run performance tests
python -c "
from tests.comprehensive_tests import TestPerformanceBenchmarks
import unittest
suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceBenchmarks)
unittest.TextTestRunner(verbosity=2).run(suite)
"
```

## Troubleshooting

### Common Installation Issues

#### Python Version Error
```bash
# Check Python version
python --version

# If version < 3.8, update Python
# Windows: Download from python.org
# macOS: brew upgrade python
# Linux: sudo apt upgrade python3
```

#### TA-Lib Installation Error
```bash
# Windows: Use pre-compiled wheel
pip install --find-links https://www.lfd.uci.edu/~gohlke/pythonlibs/ TA-Lib

# macOS: Install dependencies first
brew install ta-lib
pip install TA-Lib

# Linux: Compile from source (see detailed steps above)
```

#### Permission Errors
```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix permissions (Linux/macOS)
sudo chown -R $USER:$USER ~/.local/lib/python*/site-packages/
```

#### Memory Errors During Installation
```bash
# Increase pip cache
pip install --cache-dir /tmp/pip-cache -r requirements.txt

# Install packages individually
pip install numpy pandas
pip install scikit-learn
pip install xgboost lightgbm
```

### Platform-Specific Issues

#### Windows
- **Long Path Names**: Enable long path support in Windows
- **Antivirus**: Add project directory to antivirus exclusions
- **PowerShell**: Use PowerShell instead of Command Prompt

#### macOS
- **Xcode Tools**: Install with `xcode-select --install`
- **Homebrew**: Install if not available: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- **Permissions**: Use `sudo` carefully, prefer user installations

#### Linux
- **Build Tools**: Install with `sudo apt install build-essential`
- **Python Headers**: Install with `sudo apt install python3-dev`
- **System Libraries**: May need additional system packages

### Verification Failures

#### Import Errors
```bash
# Check installed packages
pip list

# Reinstall problematic package
pip uninstall package_name
pip install package_name

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Model Training Fails
```bash
# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')"

# Reduce dataset size for testing
python ml_system/advanced_trainer.py --test-mode

# Check disk space
df -h  # Linux/macOS
dir   # Windows
```

#### GUI Launch Fails
```bash
# Check display environment (Linux)
echo $DISPLAY

# Test matplotlib backend
python -c "import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt; print('GUI backend OK')"

# Run in debug mode
python -m pdb gui_app/main_application.py
```

### Getting Help

#### Log Analysis
```bash
# Check application logs
tail -f logs/application.log

# Check system logs
# Windows: Event Viewer
# macOS: Console app
# Linux: journalctl -f
```

#### System Information
```bash
# Collect system info for support
python -c "
import platform
import sys
import pkg_resources
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'Packages: {len(list(pkg_resources.working_set))}')
"
```

#### Support Resources
- **GitHub Issues**: Report installation problems
- **Documentation**: Check other docs in `docs/` directory
- **Community**: Join discussions for help
- **Stack Overflow**: Search for similar issues

---

Next Steps:
- [User Guide](user_guide.md) - Learn how to use the system
- [API Reference](api_reference.md) - Detailed API documentation
- [Developer Guide](developer_guide.md) - Development and customization
