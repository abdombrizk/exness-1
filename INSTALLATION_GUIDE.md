# 🛠️ Gold Trading AI - Complete Installation Guide

This comprehensive guide will help you install and set up the Gold Trading AI system with all required dependencies for optimal performance.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for data fetching and package installation

### Supported Operating Systems
- ✅ Windows 10/11
- ✅ macOS 10.15+ (Catalina or newer)
- ✅ Ubuntu 18.04+ / Debian 10+
- ✅ CentOS 7+ / RHEL 7+
- ✅ Fedora 30+

## 🚀 Quick Installation (Recommended)

### Option 1: Automated Installation Scripts

#### Windows
```batch
# Run as Administrator (optional but recommended)
install_windows.bat
```

#### macOS
```bash
chmod +x install_macos.sh
./install_macos.sh
```

#### Linux
```bash
chmod +x install_linux.sh
./install_linux.sh
```

### Option 2: Python Installation Script
```bash
python install.py
```

## 📝 Manual Installation (Step by Step)

### Step 1: Verify Python Installation

Check if Python 3.8+ is installed:
```bash
python --version
# or
python3 --version
```

If Python is not installed:
- **Windows**: Download from [python.org](https://python.org)
- **macOS**: `brew install python` or download from [python.org](https://python.org)
- **Linux**: `sudo apt install python3 python3-pip` (Ubuntu/Debian)

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3: Upgrade pip and Install Build Tools

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install build tools
pip install wheel setuptools
```

### Step 4: Install System Dependencies

#### Windows
- Install **Visual C++ Build Tools** from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Or install **Visual Studio Community** with C++ development tools

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install TA-Lib
brew install ta-lib
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y build-essential python3-dev libta-lib-dev libffi-dev libssl-dev
```

#### Linux (CentOS/RHEL)
```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-devel ta-lib-devel libffi-devel openssl-devel
```

### Step 5: Install Python Packages

#### Install Critical Packages First
```bash
pip install numpy>=1.24.0 pandas>=2.1.0 scikit-learn>=1.3.0 matplotlib>=3.8.0 requests>=2.31.0
```

#### Install TA-Lib (Technical Analysis)
```bash
# Try standard installation
pip install TA-Lib

# If that fails, try binary version (Windows)
pip install talib-binary
```

#### Install Deep Learning Frameworks
```bash
# PyTorch (CPU version for compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support (CUDA 11.8):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TensorFlow
pip install tensorflow>=2.15.0
```

#### Install All Remaining Packages
```bash
pip install -r requirements.txt
```

## 🔧 Platform-Specific Instructions

### Windows Detailed Setup

1. **Install Python 3.9+**
   - Download from [python.org](https://python.org)
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install pip"

2. **Install Visual C++ Build Tools**
   - Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Install "C++ build tools" workload

3. **Install TA-Lib**
   ```batch
   pip install TA-Lib
   ```
   If this fails, download the appropriate wheel from [Christoph Gohlke's site](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib):
   ```batch
   pip install TA_Lib-0.4.28-cp39-cp39-win_amd64.whl
   ```

### macOS Detailed Setup

1. **Install Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

2. **Install Homebrew**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Install TA-Lib**
   ```bash
   brew install ta-lib
   pip install TA-Lib
   ```

### Linux Detailed Setup

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install build dependencies
sudo apt install -y build-essential python3-dev python3-pip python3-venv

# Install TA-Lib
sudo apt install -y libta-lib-dev

# Install other dependencies
sudo apt install -y libffi-dev libssl-dev pkg-config
```

#### CentOS/RHEL
```bash
# Install development tools
sudo yum groupinstall -y "Development Tools"

# Install Python development headers
sudo yum install -y python3-devel python3-pip

# Install TA-Lib
sudo yum install -y ta-lib-devel

# Install other dependencies
sudo yum install -y libffi-devel openssl-devel
```

## 🧪 Verification and Testing

### Test Installation
```bash
# Run automated verification
python install.py

# Run system tests
python tests/test_system.py

# Run demo
python demo.py
```

### Manual Verification
```python
# Test critical imports
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import yfinance as yf
import talib
import torch
import tensorflow as tf

print("✅ All packages imported successfully!")
```

### Test Data Fetching
```python
import yfinance as yf

# Test gold data fetching
ticker = yf.Ticker("GC=F")
data = ticker.history(period="1d")
print(f"✅ Fetched {len(data)} data points")
```

### Test Technical Analysis
```python
import talib
import numpy as np

# Test TA-Lib
prices = np.random.random(100) * 100 + 2000
rsi = talib.RSI(prices)
print(f"✅ RSI calculated: {rsi[-1]:.2f}")
```

## 🚨 Troubleshooting Common Issues

### TA-Lib Installation Issues

#### Windows
**Problem**: `Microsoft Visual C++ 14.0 is required`
**Solution**: Install Visual C++ Build Tools or Visual Studio Community

**Problem**: `TA-Lib installation fails`
**Solution**: 
1. Download wheel from [Christoph Gohlke's site](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
2. Install with: `pip install downloaded_wheel.whl`

#### macOS
**Problem**: `TA-Lib installation fails`
**Solution**:
```bash
brew install ta-lib
export TA_INCLUDE_PATH="$(brew --prefix ta-lib)/include"
export TA_LIBRARY_PATH="$(brew --prefix ta-lib)/lib"
pip install TA-Lib
```

#### Linux
**Problem**: `TA-Lib installation fails`
**Solution**:
```bash
# Ubuntu/Debian
sudo apt install libta-lib-dev

# CentOS/RHEL
sudo yum install ta-lib-devel

# Then retry
pip install TA-Lib
```

### PyTorch Installation Issues

**Problem**: CUDA version mismatch
**Solution**: Install CPU version for compatibility:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Problem**: Memory issues during installation
**Solution**: Install with no cache:
```bash
pip install torch --no-cache-dir
```

### Memory Issues

**Problem**: Installation fails due to memory
**Solution**:
```bash
# Install packages one by one
pip install numpy
pip install pandas
pip install scikit-learn
# etc.
```

### Permission Issues

#### Windows
Run Command Prompt as Administrator

#### macOS/Linux
```bash
# Use --user flag
pip install --user package_name

# Or fix permissions
sudo chown -R $(whoami) ~/.local
```

## 🔄 Updating Dependencies

### Update All Packages
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Update pip
python -m pip install --upgrade pip

# Update all packages
pip install --upgrade -r requirements.txt
```

### Update Specific Packages
```bash
pip install --upgrade numpy pandas scikit-learn
pip install --upgrade torch torchvision
pip install --upgrade tensorflow
```

## 🌟 Best Practices

### Virtual Environment Management
```bash
# Always use virtual environments
python -m venv gold_trading_ai_env
source gold_trading_ai_env/bin/activate  # macOS/Linux
gold_trading_ai_env\Scripts\activate     # Windows

# Deactivate when done
deactivate
```

### Package Management
```bash
# Keep requirements.txt updated
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Check for outdated packages
pip list --outdated
```

### Performance Optimization
```bash
# For faster numpy operations
pip install intel-numpy  # Intel systems

# For GPU acceleration (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📞 Getting Help

### Common Resources
- **Python Issues**: [python.org](https://python.org)
- **PyTorch Issues**: [pytorch.org](https://pytorch.org)
- **TA-Lib Issues**: [TA-Lib Documentation](https://ta-lib.org)
- **Package Issues**: [PyPI](https://pypi.org)

### Project Support
- **GitHub Issues**: Report bugs and issues
- **Documentation**: Check `docs/` folder
- **Email**: support@aitradingsystems.com

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] System dependencies installed (build tools, TA-Lib)
- [ ] pip upgraded to latest version
- [ ] Critical packages installed (numpy, pandas, sklearn)
- [ ] TA-Lib installed and working
- [ ] PyTorch installed
- [ ] TensorFlow installed
- [ ] All requirements.txt packages installed
- [ ] Installation verified with tests
- [ ] Demo runs successfully

---

🎉 **Congratulations!** Your Gold Trading AI system is now ready to use!

Run `python demo.py` to see the system in action.
