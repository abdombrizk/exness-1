#!/usr/bin/env python3
"""
Gold Trading AI - Professional System Launcher
==============================================

Enterprise-grade launcher with setup, training, and deployment capabilities.
Provides automated installation, configuration, and system management.

Author: AI Trading Systems
Version: 2.0.0
License: MIT
"""

import sys
import os
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class GoldTradingAILauncher:
    """Comprehensive system launcher and manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_executable = sys.executable
        
    def print_banner(self):
        """Print professional banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🥇 GOLD TRADING AI - PROFESSIONAL SYSTEM                  ║
║                                                                              ║
║                    🎯 Target: >90% Accuracy ML Predictions                   ║
║                    🖥️  Bloomberg Terminal-Style Interface                     ║
║                    🗄️  Comprehensive Database Integration                     ║
║                    🧪 Full Testing & Validation Suite                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def check_python_version(self):
        """Check Python version compatibility"""
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python {version.major}.{version.minor} detected")
            print("   Minimum required: Python 3.8")
            print("   Please upgrade Python and try again")
            return False
            
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
        
    def check_dependencies(self):
        """Check if dependencies are installed"""
        print("\n📦 Checking dependencies...")
        
        required_packages = [
            'numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm',
            'matplotlib', 'yfinance', 'optuna'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - Missing")
                missing_packages.append(package)
                
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("   Run: python launch.py --install-deps")
            return False
            
        print("✅ All dependencies installed")
        return True
        
    def install_dependencies(self):
        """Install required dependencies"""
        print("\n📦 Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
            
        try:
            cmd = [self.python_executable, "-m", "pip", "install", "-r", str(requirements_file)]
            print(f"   Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
            
    def setup_directories(self):
        """Setup required directories"""
        print("\n📁 Setting up directories...")
        
        directories = [
            "ml_system/models",
            "database",
            "logs",
            "tests/reports",
            "docs/assets",
            "config",
            "backups"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
            
        print("✅ Directory structure created")
        return True
        
    def train_models(self):
        """Train ML models"""
        print("\n🤖 Training ML models...")
        print("   This may take 10-30 minutes depending on your hardware...")
        
        trainer_script = self.project_root / "src" / "core" / "models" / "trainer.py"
        
        if not trainer_script.exists():
            print("❌ ML trainer script not found")
            return False
            
        try:
            cmd = [self.python_executable, str(trainer_script)]
            print(f"   Running: {' '.join(cmd)}")
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=False, text=True)
            end_time = time.time()
            
            training_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ Model training completed in {training_time:.1f} seconds")
                return True
            else:
                print(f"❌ Model training failed")
                return False
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
            
    def initialize_database(self):
        """Initialize database with sample data"""
        print("\n🗄️  Initializing database...")
        
        db_script = self.project_root / "src" / "core" / "database" / "manager.py"
        
        if not db_script.exists():
            print("❌ Database script not found")
            return False
            
        try:
            cmd = [self.python_executable, str(db_script)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Database initialized successfully")
                return True
            else:
                print(f"❌ Database initialization failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
            
    def run_tests(self):
        """Run comprehensive test suite"""
        print("\n🧪 Running comprehensive tests...")
        
        test_script = self.project_root / "tests" / "run_all_tests.py"
        
        if not test_script.exists():
            print("❌ Test script not found")
            return False
            
        try:
            cmd = [self.python_executable, str(test_script)]
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ All tests passed")
                return True
            else:
                print("⚠️  Some tests failed - check output above")
                return False
                
        except Exception as e:
            print(f"❌ Testing error: {e}")
            return False
            
    def launch_gui(self):
        """Launch the GUI application"""
        print("\n🖥️  Launching GUI application...")
        
        gui_script = self.project_root / "src" / "gui" / "application.py"
        
        if not gui_script.exists():
            print("❌ GUI script not found")
            return False
            
        try:
            cmd = [self.python_executable, str(gui_script)]
            print(f"   Running: {' '.join(cmd)}")
            print("   GUI application starting...")
            
            # Launch GUI in background
            subprocess.Popen(cmd)
            
            print("✅ GUI application launched")
            print("   Check your screen for the application window")
            return True
            
        except Exception as e:
            print(f"❌ GUI launch error: {e}")
            return False
            
    def full_setup(self):
        """Run complete setup process"""
        print("🚀 Starting full system setup...")
        print("=" * 80)
        
        steps = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Directories", self.setup_directories),
            ("Model Training", self.train_models),
            ("Database", self.initialize_database),
            ("Testing", self.run_tests)
        ]
        
        for step_name, step_function in steps:
            print(f"\n📋 Step: {step_name}")
            if not step_function():
                print(f"❌ Setup failed at step: {step_name}")
                return False
                
        print("\n" + "=" * 80)
        print("🎉 SETUP COMPLETE!")
        print("=" * 80)
        print("✅ All components installed and tested")
        print("✅ Models trained with >90% accuracy target")
        print("✅ Database initialized with sample data")
        print("✅ All tests passed")
        print("\n🚀 Ready to launch GUI application!")
        
        return True
        
    def quick_start(self):
        """Quick start for existing installations"""
        print("⚡ Quick start mode...")
        
        # Check if models exist
        models_dir = self.project_root / "data" / "models"
        if not models_dir.exists() or not any(models_dir.glob("*.joblib")):
            print("⚠️  No trained models found")
            print("   Running model training...")
            if not self.train_models():
                return False
                
        # Launch GUI
        return self.launch_gui()
        
    def system_status(self):
        """Check system status"""
        print("📊 System Status Check")
        print("=" * 50)
        
        # Check Python
        version = sys.version_info
        print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
        
        # Check dependencies
        try:
            import numpy, pandas, sklearn, xgboost, lightgbm, matplotlib
            print("📦 Dependencies: ✅ Installed")
        except ImportError as e:
            print(f"📦 Dependencies: ❌ Missing ({e})")
            
        # Check models
        models_dir = self.project_root / "data" / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.joblib"))
            print(f"🤖 Models: ✅ {len(model_files)} found")
        else:
            print("🤖 Models: ❌ Not found")
            
        # Check database
        db_file = self.project_root / "data" / "database" / "trading_data.db"
        if db_file.exists():
            print(f"🗄️  Database: ✅ Found ({db_file.stat().st_size / 1024:.1f} KB)")
        else:
            print("🗄️  Database: ❌ Not found")
            
        # Check GUI components
        gui_script = self.project_root / "src" / "gui" / "application.py"
        if gui_script.exists():
            print("🖥️  GUI: ✅ Available")
        else:
            print("🖥️  GUI: ❌ Not found")
            
        print("=" * 50)


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Gold Trading AI - Professional System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                    # Full setup and launch
  python launch.py --quick            # Quick start (skip setup)
  python launch.py --install-deps     # Install dependencies only
  python launch.py --train            # Train models only
  python launch.py --test             # Run tests only
  python launch.py --gui              # Launch GUI only
  python launch.py --status           # Check system status
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick start (skip full setup)')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install dependencies only')
    parser.add_argument('--train', action='store_true',
                       help='Train models only')
    parser.add_argument('--test', action='store_true',
                       help='Run tests only')
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI only')
    parser.add_argument('--status', action='store_true',
                       help='Check system status')
    parser.add_argument('--setup', action='store_true',
                       help='Run full setup process')
    
    args = parser.parse_args()
    
    launcher = GoldTradingAILauncher()
    launcher.print_banner()
    
    try:
        if args.status:
            launcher.system_status()
        elif args.install_deps:
            launcher.install_dependencies()
        elif args.train:
            launcher.train_models()
        elif args.test:
            launcher.run_tests()
        elif args.gui:
            launcher.launch_gui()
        elif args.quick:
            launcher.quick_start()
        elif args.setup:
            launcher.full_setup()
        else:
            # Default: Full setup and launch
            if launcher.full_setup():
                print("\n🚀 Launching GUI application...")
                launcher.launch_gui()
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
