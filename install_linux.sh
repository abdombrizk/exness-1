#!/bin/bash
# Gold Trading AI - Linux Installation Script
# Automated setup for Linux systems (Ubuntu/Debian/CentOS/RHEL)

set -e  # Exit on any error

echo "========================================"
echo "Gold Trading AI - Linux Installation"
echo "========================================"

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo "❌ Cannot detect Linux distribution"
    exit 1
fi

echo "Detected OS: $OS"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ using your package manager"
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

# Install system dependencies based on distribution
echo ""
echo "🔧 Installing system dependencies..."

if command -v apt &> /dev/null; then
    # Ubuntu/Debian
    echo "📦 Using apt package manager..."
    sudo apt update
    sudo apt install -y build-essential
    sudo apt install -y python3-dev python3-pip python3-venv
    sudo apt install -y libta-lib-dev
    sudo apt install -y libffi-dev libssl-dev
    sudo apt install -y pkg-config
    
elif command -v yum &> /dev/null; then
    # CentOS/RHEL (older versions)
    echo "📦 Using yum package manager..."
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y python3-devel python3-pip
    sudo yum install -y ta-lib-devel
    sudo yum install -y libffi-devel openssl-devel
    
elif command -v dnf &> /dev/null; then
    # Fedora/CentOS/RHEL (newer versions)
    echo "📦 Using dnf package manager..."
    sudo dnf groupinstall -y "Development Tools"
    sudo dnf install -y python3-devel python3-pip
    sudo dnf install -y ta-lib-devel
    sudo dnf install -y libffi-devel openssl-devel
    
elif command -v pacman &> /dev/null; then
    # Arch Linux
    echo "📦 Using pacman package manager..."
    sudo pacman -S --noconfirm base-devel
    sudo pacman -S --noconfirm python python-pip
    sudo pacman -S --noconfirm ta-lib
    
else
    echo "⚠️  Unknown package manager. Please install manually:"
    echo "   - build-essential or Development Tools"
    echo "   - python3-dev or python3-devel"
    echo "   - libta-lib-dev or ta-lib-devel"
    echo "   - libffi-dev and libssl-dev"
fi

echo "✅ System dependencies installed"

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
pip install TA-Lib || {
    echo "⚠️  TA-Lib installation failed. Trying alternative..."
    pip install talib-binary || {
        echo "❌ TA-Lib installation failed"
        echo "Please ensure libta-lib-dev is installed:"
        echo "Ubuntu/Debian: sudo apt install libta-lib-dev"
        echo "CentOS/RHEL: sudo yum install ta-lib-devel"
    }
}

# Install PyTorch (CPU version for compatibility)
echo ""
echo "🤖 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

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
