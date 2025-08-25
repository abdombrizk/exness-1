#!/bin/bash
# Gold Trading AI - macOS Installation Script
# Automated setup for macOS systems

set -e  # Exit on any error

echo "========================================"
echo "Gold Trading AI - macOS Installation"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    echo "Or use Homebrew: brew install python"
    exit 1
fi

echo "Python found:"
python3 --version

# Check Python version
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" || {
    echo "❌ ERROR: Python 3.8+ required"
    exit 1
}

echo "✅ Python version is compatible"

# Check for Xcode Command Line Tools
echo ""
echo "🔧 Checking for Xcode Command Line Tools..."
if ! xcode-select -p &> /dev/null; then
    echo "⚠️  Xcode Command Line Tools not found"
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "Please complete the Xcode installation and run this script again"
    exit 1
fi

echo "✅ Xcode Command Line Tools available"

# Check for Homebrew
echo ""
echo "🍺 Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    echo "⚠️  Homebrew not found"
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "✅ Homebrew available"

# Install TA-Lib via Homebrew
echo ""
echo "📊 Installing TA-Lib via Homebrew..."
brew install ta-lib || echo "⚠️  TA-Lib may already be installed"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

echo "✅ Virtual environment created"

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install wheel and setuptools
echo ""
echo "🔧 Installing build tools..."
pip install wheel setuptools

# Install critical packages first
echo ""
echo "📊 Installing critical packages..."
pip install numpy>=1.24.0
pip install pandas>=2.1.0
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.8.0
pip install requests>=2.31.0

# Install TA-Lib Python wrapper
echo ""
echo "📊 Installing TA-Lib Python wrapper..."
pip install TA-Lib

# Install PyTorch
echo ""
echo "🤖 Installing PyTorch..."
pip install torch torchvision torchaudio

# Install TensorFlow
echo ""
echo "🧠 Installing TensorFlow..."
pip install tensorflow>=2.15.0

# Install remaining packages
echo ""
echo "📋 Installing remaining packages..."
pip install -r requirements.txt

# Run verification
echo ""
echo "🔍 Running installation verification..."
python install.py

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run: python demo.py"
echo "2. Run: python main.py"
echo "3. Test: python tests/test_system.py"
echo ""
echo "To activate environment in future:"
echo "source venv/bin/activate"
echo ""
