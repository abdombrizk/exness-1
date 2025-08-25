#!/usr/bin/env python3
"""
Gold Trading AI - Dependency Checker
Comprehensive verification of all system dependencies

Author: AI Trading Systems
Version: 1.0.0
"""

import sys
import importlib
import subprocess
import platform
import pkg_resources
from packaging import version


class DependencyChecker:
    """Comprehensive dependency checker for Gold Trading AI"""
    
    def __init__(self):
        self.system_info = {
            'platform': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': sys.version,
            'python_executable': sys.executable
        }
        
        self.results = {
            'critical': {},
            'ml_frameworks': {},
            'financial_data': {},
            'technical_analysis': {},
            'gui_visualization': {},
            'utilities': {},
            'optional': {}
        }
        
        self.errors = []
        self.warnings = []
        
    def print_system_info(self):
        """Print system information"""
        print("🖥️  SYSTEM INFORMATION")
        print("=" * 50)
        print(f"Platform: {self.system_info['platform']} {self.system_info['release']}")
        print(f"Architecture: {self.system_info['machine']}")
        print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print(f"Python Path: {self.system_info['python_executable']}")
        print("=" * 50)
        
    def check_package(self, package_name, min_version=None, import_name=None, category='utilities'):
        """Check if a package is installed and meets version requirements"""
        if import_name is None:
            import_name = package_name
            
        try:
            # Try to import the package
            module = importlib.import_module(import_name)
            
            # Get version if available
            try:
                if hasattr(module, '__version__'):
                    installed_version = module.__version__
                else:
                    # Try to get version from pkg_resources
                    installed_version = pkg_resources.get_distribution(package_name).version
            except:
                installed_version = "Unknown"
                
            # Check version requirement
            version_ok = True
            if min_version and installed_version != "Unknown":
                try:
                    version_ok = version.parse(installed_version) >= version.parse(min_version)
                except:
                    version_ok = True  # If version parsing fails, assume OK
                    
            status = "✅" if version_ok else "⚠️"
            version_info = f"{installed_version}"
            if min_version:
                version_info += f" (required: {min_version}+)"
                
            self.results[category][package_name] = {
                'status': 'OK' if version_ok else 'VERSION_WARNING',
                'version': installed_version,
                'required': min_version,
                'module': module
            }
            
            if not version_ok:
                self.warnings.append(f"{package_name} version {installed_version} < {min_version}")
                
            return True, version_info, status
            
        except ImportError as e:
            self.results[category][package_name] = {
                'status': 'MISSING',
                'error': str(e)
            }
            self.errors.append(f"{package_name} not installed: {e}")
            return False, "Not installed", "❌"
            
    def check_critical_packages(self):
        """Check critical packages required for basic functionality"""
        print("\n📦 CRITICAL PACKAGES")
        print("-" * 30)
        
        critical_packages = [
            ('numpy', '1.24.0'),
            ('pandas', '2.1.0'),
            ('scikit-learn', '1.3.0', 'sklearn'),
            ('matplotlib', '3.8.0'),
            ('requests', '2.31.0'),
            ('scipy', '1.11.0')
        ]
        
        for package_info in critical_packages:
            package_name = package_info[0]
            min_version = package_info[1]
            import_name = package_info[2] if len(package_info) > 2 else package_name
            
            success, version_info, status = self.check_package(
                package_name, min_version, import_name, 'critical'
            )
            print(f"{status} {package_name}: {version_info}")
            
    def check_ml_frameworks(self):
        """Check machine learning frameworks"""
        print("\n🤖 ML FRAMEWORKS")
        print("-" * 30)
        
        ml_packages = [
            ('torch', '2.1.0'),
            ('torchvision', '0.16.0'),
            ('tensorflow', '2.15.0'),
            ('xgboost', '2.0.0'),
            ('lightgbm', '4.1.0')
        ]
        
        for package_info in ml_packages:
            package_name = package_info[0]
            min_version = package_info[1]
            
            success, version_info, status = self.check_package(
                package_name, min_version, package_name, 'ml_frameworks'
            )
            print(f"{status} {package_name}: {version_info}")
            
            # Special checks for PyTorch and TensorFlow
            if success and package_name == 'torch':
                self._check_pytorch_features()
            elif success and package_name == 'tensorflow':
                self._check_tensorflow_features()
                
    def check_financial_data_packages(self):
        """Check financial data packages"""
        print("\n💰 FINANCIAL DATA")
        print("-" * 30)
        
        financial_packages = [
            ('yfinance', '0.2.25'),
            ('alpha-vantage', '2.3.1', 'alpha_vantage'),
            ('fredapi', '0.5.1'),
            ('requests', '2.31.0')
        ]
        
        for package_info in financial_packages:
            package_name = package_info[0]
            min_version = package_info[1]
            import_name = package_info[2] if len(package_info) > 2 else package_name
            
            success, version_info, status = self.check_package(
                package_name, min_version, import_name, 'financial_data'
            )
            print(f"{status} {package_name}: {version_info}")
            
    def check_technical_analysis_packages(self):
        """Check technical analysis packages"""
        print("\n📊 TECHNICAL ANALYSIS")
        print("-" * 30)
        
        ta_packages = [
            ('TA-Lib', '0.4.28', 'talib'),
            ('pandas-ta', '0.3.14', 'pandas_ta')
        ]
        
        for package_info in ta_packages:
            package_name = package_info[0]
            min_version = package_info[1]
            import_name = package_info[2] if len(package_info) > 2 else package_name
            
            success, version_info, status = self.check_package(
                package_name, min_version, import_name, 'technical_analysis'
            )
            print(f"{status} {package_name}: {version_info}")
            
            # Test TA-Lib functionality
            if success and package_name == 'TA-Lib':
                self._test_talib_functionality()
                
    def check_gui_visualization_packages(self):
        """Check GUI and visualization packages"""
        print("\n🖼️  GUI & VISUALIZATION")
        print("-" * 30)
        
        gui_packages = [
            ('matplotlib', '3.8.0'),
            ('plotly', '5.17.0'),
            ('seaborn', '0.13.0'),
            ('tkinter', None)  # Built-in, no version check
        ]
        
        for package_info in gui_packages:
            package_name = package_info[0]
            min_version = package_info[1]
            
            success, version_info, status = self.check_package(
                package_name, min_version, package_name, 'gui_visualization'
            )
            print(f"{status} {package_name}: {version_info}")
            
    def check_utility_packages(self):
        """Check utility packages"""
        print("\n🔧 UTILITIES")
        print("-" * 30)
        
        utility_packages = [
            ('python-dateutil', '2.8.2', 'dateutil'),
            ('pytz', '2023.3'),
            ('tqdm', '4.66.0'),
            ('joblib', '1.3.0'),
            ('sqlite3', None)  # Built-in
        ]
        
        for package_info in utility_packages:
            package_name = package_info[0]
            min_version = package_info[1]
            import_name = package_info[2] if len(package_info) > 2 else package_name
            
            success, version_info, status = self.check_package(
                package_name, min_version, import_name, 'utilities'
            )
            print(f"{status} {package_name}: {version_info}")
            
    def check_optional_packages(self):
        """Check optional packages"""
        print("\n⭐ OPTIONAL PACKAGES")
        print("-" * 30)
        
        optional_packages = [
            ('nltk', '3.8.1'),
            ('textblob', '0.17.1'),
            ('transformers', '4.36.0'),
            ('beautifulsoup4', '4.12.0', 'bs4'),
            ('psutil', '5.9.0')
        ]
        
        for package_info in optional_packages:
            package_name = package_info[0]
            min_version = package_info[1]
            import_name = package_info[2] if len(package_info) > 2 else package_name
            
            success, version_info, status = self.check_package(
                package_name, min_version, import_name, 'optional'
            )
            print(f"{status} {package_name}: {version_info}")
            
    def _check_pytorch_features(self):
        """Check PyTorch specific features"""
        try:
            import torch
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                print(f"   🚀 CUDA: Available (v{cuda_version}, {gpu_count} GPU(s))")
            else:
                print(f"   💻 CUDA: Not available (CPU only)")
                
            # Check MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print(f"   🍎 MPS: Available (Apple Silicon)")
                
        except Exception as e:
            print(f"   ⚠️  PyTorch feature check failed: {e}")
            
    def _check_tensorflow_features(self):
        """Check TensorFlow specific features"""
        try:
            import tensorflow as tf
            
            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"   🚀 GPU: {len(gpus)} device(s) available")
            else:
                print(f"   💻 GPU: Not available (CPU only)")
                
        except Exception as e:
            print(f"   ⚠️  TensorFlow feature check failed: {e}")
            
    def _test_talib_functionality(self):
        """Test TA-Lib functionality"""
        try:
            import talib
            import numpy as np
            
            # Test basic indicator calculation
            prices = np.random.random(100) * 100 + 2000
            rsi = talib.RSI(prices)
            
            if not np.isnan(rsi[-1]):
                print(f"   ✅ TA-Lib functional (RSI test passed)")
            else:
                print(f"   ⚠️  TA-Lib test returned NaN")
                
        except Exception as e:
            print(f"   ❌ TA-Lib functionality test failed: {e}")
            
    def test_data_fetching(self):
        """Test data fetching capabilities"""
        print("\n🌐 DATA FETCHING TEST")
        print("-" * 30)
        
        try:
            import yfinance as yf
            
            # Test gold data fetching
            ticker = yf.Ticker("GC=F")
            data = ticker.history(period="1d")
            
            if not data.empty:
                print(f"✅ Yahoo Finance: Fetched {len(data)} gold price records")
            else:
                print(f"⚠️  Yahoo Finance: No data returned")
                
        except Exception as e:
            print(f"❌ Yahoo Finance test failed: {e}")
            
        # Test other data sources
        try:
            import requests
            response = requests.get("https://httpbin.org/get", timeout=5)
            if response.status_code == 200:
                print(f"✅ HTTP Requests: Working")
            else:
                print(f"⚠️  HTTP Requests: Status {response.status_code}")
        except Exception as e:
            print(f"❌ HTTP Requests test failed: {e}")
            
    def generate_summary(self):
        """Generate installation summary"""
        print("\n" + "=" * 60)
        print("📊 DEPENDENCY CHECK SUMMARY")
        print("=" * 60)
        
        total_packages = 0
        working_packages = 0
        
        for category, packages in self.results.items():
            category_working = 0
            category_total = len(packages)
            total_packages += category_total
            
            for package, info in packages.items():
                if info['status'] in ['OK', 'VERSION_WARNING']:
                    category_working += 1
                    working_packages += 1
                    
            if category_total > 0:
                success_rate = (category_working / category_total) * 100
                print(f"{category.replace('_', ' ').title()}: {category_working}/{category_total} ({success_rate:.0f}%)")
                
        overall_success_rate = (working_packages / total_packages) * 100 if total_packages > 0 else 0
        
        print(f"\nOverall: {working_packages}/{total_packages} ({overall_success_rate:.0f}%)")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
                
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
                
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if overall_success_rate >= 90:
            print("   🎉 Excellent! Your system is ready for Gold Trading AI")
        elif overall_success_rate >= 75:
            print("   ✅ Good! Address warnings for optimal performance")
        elif overall_success_rate >= 50:
            print("   ⚠️  Fair. Install missing critical packages")
        else:
            print("   ❌ Poor. Run installation script: python install.py")
            
        print("=" * 60)
        
    def run_full_check(self):
        """Run complete dependency check"""
        self.print_system_info()
        self.check_critical_packages()
        self.check_ml_frameworks()
        self.check_financial_data_packages()
        self.check_technical_analysis_packages()
        self.check_gui_visualization_packages()
        self.check_utility_packages()
        self.check_optional_packages()
        self.test_data_fetching()
        self.generate_summary()


def main():
    """Main function"""
    print("🥇 Gold Trading AI - Dependency Checker")
    print("=" * 60)
    
    checker = DependencyChecker()
    checker.run_full_check()
    
    # Return exit code based on critical errors
    critical_errors = [error for error in checker.errors if any(
        pkg in error for pkg in ['numpy', 'pandas', 'sklearn', 'matplotlib', 'requests']
    )]
    
    return 0 if not critical_errors else 1


if __name__ == "__main__":
    sys.exit(main())
