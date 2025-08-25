#!/usr/bin/env python3
"""
Gold Trading AI - Setup Script
==============================

Professional setup script for the Gold Trading AI system.
Handles installation, dependencies, and package configuration.

Author: AI Trading Systems
Version: 2.0.0
License: MIT
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required")

# Get project root directory
PROJECT_ROOT = Path(__file__).parent

# Read long description from README
def read_long_description():
    """Read long description from README.md"""
    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Professional machine learning system for gold trading prediction"

# Read requirements from requirements.txt
def read_requirements(filename="requirements.txt"):
    """Read requirements from file"""
    req_path = PROJECT_ROOT / filename
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Read version from src/__init__.py
def read_version():
    """Read version from package __init__.py"""
    init_path = PROJECT_ROOT / "src" / "__init__.py"
    if init_path.exists():
        with open(init_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "2.0.0"

# Package configuration
setup(
    # Basic package information
    name="gold-trading-ai",
    version=read_version(),
    description="Professional machine learning system for gold trading prediction",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    
    # Author and contact information
    author="AI Trading Systems",
    author_email="support@goldtradingai.com",
    maintainer="AI Trading Systems",
    maintainer_email="support@goldtradingai.com",
    
    # URLs and links
    url="https://github.com/your-repo/gold-trading-ai",
    project_urls={
        "Documentation": "https://github.com/your-repo/gold-trading-ai/docs",
        "Source": "https://github.com/your-repo/gold-trading-ai",
        "Tracker": "https://github.com/your-repo/gold-trading-ai/issues",
    },
    
    # License and classification
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Package discovery and structure
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": [
            "*.yaml",
            "*.yml", 
            "*.json",
            "*.txt",
            "*.md",
            "*.cfg",
            "*.ini"
        ],
    },
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
            "pytest-xdist>=2.5.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "performance": [
            "numba>=0.56.0",
            "cython>=0.29.0",
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "gold-trading-ai=src.cli:main",
            "gta-train=src.core.models.trainer:main",
            "gta-gui=src.gui.application:main",
            "gta-setup=scripts.setup:main",
        ],
    },
    
    # Keywords for package discovery
    keywords=[
        "trading",
        "gold",
        "machine-learning",
        "finance",
        "prediction",
        "ai",
        "ensemble",
        "technical-analysis",
        "bloomberg-terminal",
        "gui"
    ],
    
    # Zip safety
    zip_safe=False,
    
    # Platform compatibility
    platforms=["any"],
    
    # Additional metadata
    options={
        "bdist_wheel": {
            "universal": False,
        },
    },
)

# Post-installation setup
def post_install():
    """Run post-installation setup"""
    try:
        # Create necessary directories
        directories = [
            "data/raw/market_data",
            "data/processed/features", 
            "data/models",
            "data/database",
            "data/backups",
            "logs",
            "outputs/reports",
            "outputs/charts",
            "outputs/exports",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        print("✅ Directory structure created")
        
        # Copy configuration templates
        config_templates = [
            ("config/templates/.env.template", ".env.example"),
            ("config/templates/settings.template.yaml", "config/settings.example.yaml"),
        ]
        
        for src, dst in config_templates:
            src_path = Path(src)
            dst_path = Path(dst)
            
            if src_path.exists() and not dst_path.exists():
                import shutil
                shutil.copy2(src_path, dst_path)
                print(f"✅ Created {dst}")
                
        print("✅ Post-installation setup complete")
        
    except Exception as e:
        print(f"⚠️  Post-installation setup warning: {e}")

if __name__ == "__main__":
    # Run post-installation setup if installing
    if "install" in sys.argv:
        import atexit
        atexit.register(post_install)
