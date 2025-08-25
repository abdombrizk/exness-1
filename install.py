#!/usr/bin/env python3
"""
Gold Trading AI - Automated Installation Script
Comprehensive installation and setup for all dependencies

Author: AI Trading Systems
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
import importlib
import pkg_resources
from pathlib import Path


class GoldTradingAIInstaller:
    """Automated installer for Gold Trading AI system"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.errors = []
        self.warnings = []
        
        print("🥇 Gold Trading AI - Automated Installation")
        print("=" * 60)
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version}")
        print(f"Architecture: {platform.machine()}")
        print("=" * 60)
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("\n📋 Checking Python version...")
        
        if self.python_version < (3, 8):
            self.errors.append("Python 3.8+ required. Current version: {}.{}.{}".format(
                self.python_version.major, self.python_version.minor, self.python_version.micro
            ))
            return False
        elif self.python_version < (3, 9):
            self.warnings.append("Python 3.9+ recommended for optimal performance")
            
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor} is compatible")
        return True
        
    def check_pip(self):
        """Check if pip is available and up to date"""
        print("\n📦 Checking pip installation...")
        
        try:
            import pip
            print("✅ pip is available")
            
            # Upgrade pip
            print("🔄 Upgrading pip to latest version...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            print("✅ pip upgraded successfully")
            return True
            
        except ImportError:
            self.errors.append("pip is not installed. Please install pip first.")
            return False
        except subprocess.CalledProcessError as e:
            self.warnings.append(f"Could not upgrade pip: {e}")
            return True
            
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        print("\n🔧 Installing system dependencies...")
        
        if self.system == "windows":
            self._install_windows_dependencies()
        elif self.system == "darwin":  # macOS
            self._install_macos_dependencies()
        elif self.system == "linux":
            self._install_linux_dependencies()
        else:
            self.warnings.append(f"Unknown system: {self.system}")
            
    def _install_windows_dependencies(self):
        """Install Windows-specific dependencies"""
        print("🪟 Windows system detected")
        
        # Check for Visual C++ Build Tools
        print("   Checking for Visual C++ Build Tools...")
        try:
            # Try to import a package that requires compilation
            import distutils.msvccompiler
            print("   ✅ Visual C++ Build Tools available")
        except:
            self.warnings.append(
                "Visual C++ Build Tools may be missing. "
                "Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            )
            
    def _install_macos_dependencies(self):
        """Install macOS-specific dependencies"""
        print("🍎 macOS system detected")
        
        # Check for Xcode Command Line Tools
        print("   Checking for Xcode Command Line Tools...")
        try:
            result = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ Xcode Command Line Tools available")
            else:
                print("   ⚠️  Installing Xcode Command Line Tools...")
                subprocess.run(["xcode-select", "--install"])
        except FileNotFoundError:
            self.warnings.append("Xcode Command Line Tools not found. Run: xcode-select --install")
            
        # Check for Homebrew (for TA-Lib)
        print("   Checking for Homebrew...")
        try:
            subprocess.run(["brew", "--version"], capture_output=True, check=True)
            print("   ✅ Homebrew available")
            
            # Install TA-Lib via Homebrew
            print("   📊 Installing TA-Lib via Homebrew...")
            subprocess.run(["brew", "install", "ta-lib"], capture_output=True)
            print("   ✅ TA-Lib installed")
            
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.warnings.append(
                "Homebrew not found. Install from: https://brew.sh/ "
                "Then run: brew install ta-lib"
            )
            
    def _install_linux_dependencies(self):
        """Install Linux-specific dependencies"""
        print("🐧 Linux system detected")
        
        # Try to install build essentials and TA-Lib
        try:
            # Check for apt (Ubuntu/Debian)
            subprocess.run(["apt", "--version"], capture_output=True, check=True)
            print("   📦 Installing build dependencies...")
            
            commands = [
                ["sudo", "apt", "update"],
                ["sudo", "apt", "install", "-y", "build-essential"],
                ["sudo", "apt", "install", "-y", "python3-dev"],
                ["sudo", "apt", "install", "-y", "libta-lib-dev"],
                ["sudo", "apt", "install", "-y", "libffi-dev"],
                ["sudo", "apt", "install", "-y", "libssl-dev"]
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd, capture_output=True, check=True)
                    print(f"   ✅ {' '.join(cmd[2:])}")
                except subprocess.CalledProcessError:
                    self.warnings.append(f"Could not run: {' '.join(cmd)}")
                    
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Try yum (CentOS/RHEL)
            try:
                subprocess.run(["yum", "--version"], capture_output=True, check=True)
                print("   📦 Installing build dependencies (yum)...")
                
                commands = [
                    ["sudo", "yum", "groupinstall", "-y", "Development Tools"],
                    ["sudo", "yum", "install", "-y", "python3-devel"],
                    ["sudo", "yum", "install", "-y", "ta-lib-devel"]
                ]
                
                for cmd in commands:
                    try:
                        subprocess.run(cmd, capture_output=True, check=True)
                        print(f"   ✅ {' '.join(cmd[2:])}")
                    except subprocess.CalledProcessError:
                        self.warnings.append(f"Could not run: {' '.join(cmd)}")
                        
            except (FileNotFoundError, subprocess.CalledProcessError):
                self.warnings.append(
                    "Could not detect package manager. "
                    "Please install build-essential, python3-dev, and libta-lib-dev manually"
                )
                
    def install_python_packages(self):
        """Install Python packages from requirements.txt"""
        print("\n📦 Installing Python packages...")
        
        # Install packages in stages for better error handling
        critical_packages = [
            "numpy>=1.24.0,<1.27.0",
            "pandas>=2.1.0,<2.2.0",
            "scikit-learn>=1.3.0,<1.4.0",
            "matplotlib>=3.8.0,<3.9.0",
            "requests>=2.31.0,<3.0.0"
        ]
        
        ml_packages = [
            "torch>=2.1.0,<2.3.0",
            "torchvision>=0.16.0,<0.18.0",
            "tensorflow>=2.15.0,<2.16.0",
            "xgboost>=2.0.0,<2.1.0",
            "lightgbm>=4.1.0,<4.2.0"
        ]
        
        financial_packages = [
            "yfinance>=0.2.25",
            "alpha-vantage>=2.3.1",
            "fredapi>=0.5.1",
            "TA-Lib>=0.4.28"
        ]
        
        # Install critical packages first
        print("   📊 Installing critical packages...")
        self._install_package_list(critical_packages)
        
        # Install ML packages
        print("   🤖 Installing ML packages...")
        self._install_package_list(ml_packages)
        
        # Install financial packages
        print("   💰 Installing financial packages...")
        self._install_package_list(financial_packages)
        
        # Install remaining packages from requirements.txt
        print("   📋 Installing remaining packages...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("   ✅ All packages installed successfully")
        except subprocess.CalledProcessError as e:
            self.warnings.append(f"Some packages failed to install: {e}")
            
    def _install_package_list(self, packages):
        """Install a list of packages"""
        for package in packages:
            try:
                print(f"      Installing {package.split('>=')[0]}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True)
                print(f"      ✅ {package.split('>=')[0]} installed")
            except subprocess.CalledProcessError as e:
                self.warnings.append(f"Failed to install {package}: {e}")
                
    def install_talib_special(self):
        """Special installation for TA-Lib"""
        print("\n📊 Installing TA-Lib (Technical Analysis Library)...")
        
        try:
            # Try to import TA-Lib first
            import talib
            print("✅ TA-Lib already installed and working")
            return True
        except ImportError:
            pass
            
        # Platform-specific TA-Lib installation
        if self.system == "windows":
            print("   🪟 Installing TA-Lib for Windows...")
            try:
                # Try binary wheel first
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "TA-Lib"
                ])
                print("   ✅ TA-Lib installed via pip")
                return True
            except subprocess.CalledProcessError:
                # Try alternative binary
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "talib-binary"
                    ])
                    print("   ✅ TA-Lib installed via talib-binary")
                    return True
                except subprocess.CalledProcessError:
                    self.errors.append(
                        "TA-Lib installation failed. "
                        "Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"
                    )
                    return False
                    
        elif self.system == "darwin":
            print("   🍎 Installing TA-Lib for macOS...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "TA-Lib"
                ])
                print("   ✅ TA-Lib installed")
                return True
            except subprocess.CalledProcessError:
                self.errors.append(
                    "TA-Lib installation failed. "
                    "Run: brew install ta-lib, then pip install TA-Lib"
                )
                return False
                
        elif self.system == "linux":
            print("   🐧 Installing TA-Lib for Linux...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "TA-Lib"
                ])
                print("   ✅ TA-Lib installed")
                return True
            except subprocess.CalledProcessError:
                self.errors.append(
                    "TA-Lib installation failed. "
                    "Install libta-lib-dev first: sudo apt-get install libta-lib-dev"
                )
                return False
                
    def verify_installation(self):
        """Verify that all critical packages are working"""
        print("\n🔍 Verifying installation...")
        
        critical_imports = [
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("sklearn", "Scikit-learn"),
            ("matplotlib", "Matplotlib"),
            ("yfinance", "Yahoo Finance"),
            ("requests", "Requests")
        ]
        
        optional_imports = [
            ("torch", "PyTorch"),
            ("tensorflow", "TensorFlow"),
            ("talib", "TA-Lib"),
            ("xgboost", "XGBoost"),
            ("lightgbm", "LightGBM")
        ]
        
        # Test critical imports
        print("   Testing critical packages...")
        for module, name in critical_imports:
            try:
                importlib.import_module(module)
                print(f"   ✅ {name}")
            except ImportError as e:
                self.errors.append(f"{name} import failed: {e}")
                
        # Test optional imports
        print("   Testing optional packages...")
        for module, name in optional_imports:
            try:
                importlib.import_module(module)
                print(f"   ✅ {name}")
            except ImportError as e:
                self.warnings.append(f"{name} not available: {e}")
                
    def test_functionality(self):
        """Test basic functionality"""
        print("\n🧪 Testing basic functionality...")
        
        try:
            # Test data fetching
            print("   Testing data fetching...")
            import yfinance as yf
            ticker = yf.Ticker("GC=F")
            data = ticker.history(period="1d")
            if not data.empty:
                print("   ✅ Data fetching works")
            else:
                self.warnings.append("Data fetching returned empty results")
                
        except Exception as e:
            self.warnings.append(f"Data fetching test failed: {e}")
            
        try:
            # Test technical analysis
            print("   Testing technical analysis...")
            import talib
            import numpy as np
            
            # Create sample data
            prices = np.random.random(100) * 100 + 2000
            rsi = talib.RSI(prices)
            
            if not np.isnan(rsi[-1]):
                print("   ✅ Technical analysis works")
            else:
                self.warnings.append("Technical analysis returned NaN")
                
        except Exception as e:
            self.warnings.append(f"Technical analysis test failed: {e}")
            
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating project directories...")
        
        directories = [
            "models/trained_models",
            "database",
            "logs",
            "config",
            "tests",
            "docs",
            "data/cache"
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"   ✅ {directory}")
            except Exception as e:
                self.warnings.append(f"Could not create {directory}: {e}")
                
    def run_installation(self):
        """Run the complete installation process"""
        print("\n🚀 Starting installation process...\n")
        
        # Check prerequisites
        if not self.check_python_version():
            self.print_summary()
            return False
            
        if not self.check_pip():
            self.print_summary()
            return False
            
        # Install dependencies
        self.install_system_dependencies()
        self.create_directories()
        self.install_python_packages()
        self.install_talib_special()
        
        # Verify installation
        self.verify_installation()
        self.test_functionality()
        
        # Print summary
        self.print_summary()
        
        return len(self.errors) == 0
        
    def print_summary(self):
        """Print installation summary"""
        print("\n" + "=" * 60)
        print("📊 INSTALLATION SUMMARY")
        print("=" * 60)
        
        if not self.errors and not self.warnings:
            print("🎉 INSTALLATION SUCCESSFUL!")
            print("✅ All packages installed and verified")
            print("🚀 Gold Trading AI is ready to use!")
            
        elif not self.errors:
            print("✅ INSTALLATION COMPLETED WITH WARNINGS")
            print(f"⚠️  {len(self.warnings)} warnings found:")
            for warning in self.warnings:
                print(f"   • {warning}")
                
        else:
            print("❌ INSTALLATION FAILED")
            print(f"❌ {len(self.errors)} errors found:")
            for error in self.errors:
                print(f"   • {error}")
                
            if self.warnings:
                print(f"\n⚠️  {len(self.warnings)} warnings:")
                for warning in self.warnings:
                    print(f"   • {warning}")
                    
        print("\n" + "=" * 60)
        print("📚 Next Steps:")
        print("1. Run: python demo.py")
        print("2. Run: python main.py")
        print("3. Check: python tests/test_system.py")
        print("=" * 60)


def main():
    """Main installation function"""
    installer = GoldTradingAIInstaller()
    success = installer.run_installation()
    
    if success:
        print("\n🎉 Installation completed successfully!")
        print("Run 'python demo.py' to test the system.")
    else:
        print("\n❌ Installation encountered errors.")
        print("Please check the error messages above and resolve them.")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
