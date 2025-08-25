@echo off
REM Gold Trading AI - Windows Installation Script
REM Automated setup for Windows systems

echo ========================================
echo Gold Trading AI - Windows Installation
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found:
python --version

REM Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.8+ required
    pause
    exit /b 1
)

echo ✅ Python version is compatible

REM Create virtual environment
echo.
echo 📦 Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created

REM Activate virtual environment
echo.
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install wheel and setuptools
echo.
echo 🔧 Installing build tools...
pip install wheel setuptools

REM Install critical packages first
echo.
echo 📊 Installing critical packages...
pip install numpy>=1.24.0
pip install pandas>=2.1.0
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.8.0
pip install requests>=2.31.0

REM Install TA-Lib (Windows specific)
echo.
echo 📊 Installing TA-Lib for Windows...
pip install TA-Lib
if errorlevel 1 (
    echo ⚠️  TA-Lib pip install failed, trying alternative...
    pip install talib-binary
    if errorlevel 1 (
        echo ❌ TA-Lib installation failed
        echo Please download TA-Lib wheel from:
        echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
        echo Then install with: pip install downloaded_file.whl
    )
)

REM Install PyTorch (CPU version for compatibility)
echo.
echo 🤖 Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install TensorFlow
echo.
echo 🧠 Installing TensorFlow...
pip install tensorflow>=2.15.0

REM Install remaining packages
echo.
echo 📋 Installing remaining packages...
pip install -r requirements.txt

REM Run verification
echo.
echo 🔍 Running installation verification...
python install.py

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run: python demo.py
echo 2. Run: python main.py
echo 3. Test: python tests\test_system.py
echo.
echo To activate environment in future:
echo call venv\Scripts\activate.bat
echo.
pause
