#!/usr/bin/env python3
"""
Install Missing Dependencies for Gold Trading AI
Focused installation of missing critical packages

Author: AI Trading Systems
Version: 1.0.0
"""

import subprocess
import sys
import platform


def install_package(package_name, pip_name=None, extra_args=None):
    """Install a package with error handling"""
    if pip_name is None:
        pip_name = package_name
        
    try:
        print(f"📦 Installing {package_name}...")
        
        cmd = [sys.executable, "-m", "pip", "install", pip_name]
        if extra_args:
            cmd.extend(extra_args)
            
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {package_name} installed successfully")
            return True
        else:
            print(f"❌ {package_name} installation failed:")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {package_name} installation timed out")
        return False
    except Exception as e:
        print(f"❌ {package_name} installation error: {e}")
        return False


def install_talib_windows():
    """Special installation for TA-Lib on Windows"""
    print("📊 Installing TA-Lib for Windows...")
    
    # Try standard installation first
    if install_package("TA-Lib"):
        return True
        
    # Try binary wheel
    print("   Trying talib-binary...")
    if install_package("TA-Lib (binary)", "talib-binary"):
        return True
        
    # Try specific wheel for Python 3.12
    print("   Trying specific wheel...")
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    architecture = "win_amd64" if platform.machine() == "AMD64" else "win32"
    
    wheel_url = f"https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.28-cp{python_version}-cp{python_version}-{architecture}.whl"
    
    try:
        if install_package("TA-Lib (wheel)", wheel_url):
            return True
    except:
        pass
        
    print("❌ TA-Lib installation failed. Manual installation required:")
    print("   1. Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
    print("   2. Install with: pip install downloaded_wheel.whl")
    return False


def main():
    """Install missing packages"""
    print("🥇 Gold Trading AI - Installing Missing Dependencies")
    print("=" * 60)
    
    system = platform.system().lower()
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("=" * 60)
    
    # List of missing packages to install
    packages_to_install = [
        # ML Frameworks
        ("XGBoost", "xgboost>=2.0.0"),
        ("LightGBM", "lightgbm>=4.1.0"),
        
        # Financial Data APIs
        ("Alpha Vantage", "alpha-vantage>=2.3.1"),
        ("FRED API", "fredapi>=0.5.1"),
        
        # Technical Analysis
        ("pandas-ta", "pandas-ta>=0.3.14"),
        
        # NLP/Transformers
        ("Transformers", "transformers>=4.36.0"),
        ("Tokenizers", "tokenizers>=0.15.0"),
    ]
    
    successful_installs = 0
    total_packages = len(packages_to_install)
    
    # Install regular packages
    for package_name, pip_name in packages_to_install:
        if install_package(package_name, pip_name):
            successful_installs += 1
            
    # Special handling for TA-Lib
    print("\n📊 Installing TA-Lib (Technical Analysis Library)...")
    if system == "windows":
        if install_talib_windows():
            successful_installs += 1
            total_packages += 1
    else:
        if install_package("TA-Lib", "TA-Lib>=0.4.28"):
            successful_installs += 1
            total_packages += 1
            
    # Install additional useful packages
    additional_packages = [
        ("Statsmodels", "statsmodels>=0.14.0"),
        ("Plotly Dash", "dash>=2.14.0"),
        ("Loguru", "loguru>=0.7.0"),
        ("APScheduler", "APScheduler>=3.10.0"),
        ("Memory Profiler", "memory-profiler>=0.61.0"),
    ]
    
    print("\n📦 Installing additional useful packages...")
    for package_name, pip_name in additional_packages:
        if install_package(package_name, pip_name):
            successful_installs += 1
        total_packages += 1
        
    # Summary
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    
    success_rate = (successful_installs / total_packages) * 100
    print(f"Successfully installed: {successful_installs}/{total_packages} ({success_rate:.0f}%)")
    
    if success_rate >= 90:
        print("🎉 Excellent! Most packages installed successfully")
    elif success_rate >= 70:
        print("✅ Good! Most critical packages are now available")
    else:
        print("⚠️  Some packages failed to install. Check error messages above.")
        
    print("\n💡 Next Steps:")
    print("1. Run: python check_dependencies.py")
    print("2. Run: python demo.py")
    print("3. Run: python main.py")
    
    print("=" * 60)
    
    return 0 if success_rate >= 70 else 1


if __name__ == "__main__":
    sys.exit(main())
