# Scripts Directory

This directory contains utility scripts for setup, maintenance, and automation of the Gold Trading AI system.

## Directory Structure

```
scripts/
├── setup/                      # Setup and installation scripts
│   ├── install_dependencies.py # Automated dependency installation
│   ├── setup_database.py      # Database initialization
│   ├── create_directories.py  # Directory structure creation
│   └── verify_installation.py # Installation verification
├── data/                       # Data management scripts
│   ├── collect_data.py        # Data collection automation
│   ├── process_data.py        # Data processing pipeline
│   ├── backup_data.py         # Data backup automation
│   └── clean_data.py          # Data cleanup and maintenance
├── models/                     # Model management scripts
│   ├── train_models.py        # Model training automation
│   ├── evaluate_models.py     # Model evaluation and comparison
│   ├── deploy_models.py       # Model deployment
│   └── benchmark_models.py    # Model performance benchmarking
├── maintenance/                # System maintenance scripts
│   ├── health_check.py        # System health monitoring
│   ├── performance_monitor.py # Performance monitoring
│   ├── log_analyzer.py        # Log analysis and reporting
│   └── cleanup_system.py      # System cleanup and optimization
├── deployment/                 # Deployment and distribution scripts
│   ├── build_package.py       # Package building
│   ├── create_installer.py    # Installer creation
│   ├── deploy_production.py   # Production deployment
│   └── update_system.py       # System updates
└── utilities/                  # General utility scripts
    ├── config_manager.py      # Configuration management
    ├── backup_manager.py      # Backup management
    ├── report_generator.py    # Report generation
    └── system_info.py         # System information gathering
```

## Setup Scripts (`scripts/setup/`)

### Install Dependencies (`install_dependencies.py`)
Automated installation of all required dependencies with platform detection and error handling.

```bash
# Install all dependencies
python scripts/setup/install_dependencies.py

# Install with specific options
python scripts/setup/install_dependencies.py --dev --performance
```

### Setup Database (`setup_database.py`)
Initialize database schema and create sample data.

```bash
# Initialize database
python scripts/setup/setup_database.py

# Initialize with sample data
python scripts/setup/setup_database.py --sample-data
```

### Create Directories (`create_directories.py`)
Create the complete directory structure with proper permissions.

```bash
# Create directory structure
python scripts/setup/create_directories.py

# Create with specific permissions
python scripts/setup/create_directories.py --mode 755
```

### Verify Installation (`verify_installation.py`)
Comprehensive installation verification and system check.

```bash
# Verify installation
python scripts/setup/verify_installation.py

# Detailed verification report
python scripts/setup/verify_installation.py --detailed
```

## Data Management Scripts (`scripts/data/`)

### Collect Data (`collect_data.py`)
Automated data collection from multiple sources with scheduling support.

```bash
# Collect current data
python scripts/data/collect_data.py

# Collect historical data
python scripts/data/collect_data.py --historical --period 1y

# Schedule data collection
python scripts/data/collect_data.py --schedule daily
```

### Process Data (`process_data.py`)
Data processing pipeline with feature engineering and validation.

```bash
# Process all data
python scripts/data/process_data.py

# Process specific dataset
python scripts/data/process_data.py --input data/raw/gold_prices.csv

# Process with custom configuration
python scripts/data/process_data.py --config config/processing.yaml
```

### Backup Data (`backup_data.py`)
Automated data backup with compression and cloud storage support.

```bash
# Create backup
python scripts/data/backup_data.py

# Create compressed backup
python scripts/data/backup_data.py --compress

# Backup to cloud storage
python scripts/data/backup_data.py --cloud s3://backup-bucket
```

### Clean Data (`clean_data.py`)
Data cleanup and maintenance with configurable retention policies.

```bash
# Clean old data
python scripts/data/clean_data.py

# Clean with custom retention
python scripts/data/clean_data.py --retention-days 180

# Dry run (preview only)
python scripts/data/clean_data.py --dry-run
```

## Model Management Scripts (`scripts/models/`)

### Train Models (`train_models.py`)
Automated model training with hyperparameter optimization and validation.

```bash
# Train all models
python scripts/models/train_models.py

# Train specific model
python scripts/models/train_models.py --model lightgbm

# Train with custom target accuracy
python scripts/models/train_models.py --target-accuracy 0.95
```

### Evaluate Models (`evaluate_models.py`)
Model evaluation and comparison with detailed performance metrics.

```bash
# Evaluate all models
python scripts/models/evaluate_models.py

# Compare models
python scripts/models/evaluate_models.py --compare

# Generate evaluation report
python scripts/models/evaluate_models.py --report outputs/model_evaluation.pdf
```

### Deploy Models (`deploy_models.py`)
Model deployment with version management and rollback support.

```bash
# Deploy best model
python scripts/models/deploy_models.py

# Deploy specific model
python scripts/models/deploy_models.py --model-id model_20240115_143022

# Deploy with rollback capability
python scripts/models/deploy_models.py --enable-rollback
```

### Benchmark Models (`benchmark_models.py`)
Performance benchmarking and optimization analysis.

```bash
# Run benchmarks
python scripts/models/benchmark_models.py

# Benchmark specific operations
python scripts/models/benchmark_models.py --operations training,prediction

# Save benchmark results
python scripts/models/benchmark_models.py --output outputs/benchmarks.json
```

## Maintenance Scripts (`scripts/maintenance/`)

### Health Check (`health_check.py`)
Comprehensive system health monitoring and alerting.

```bash
# Run health check
python scripts/maintenance/health_check.py

# Continuous monitoring
python scripts/maintenance/health_check.py --monitor --interval 300

# Generate health report
python scripts/maintenance/health_check.py --report outputs/health_report.html
```

### Performance Monitor (`performance_monitor.py`)
Real-time performance monitoring with metrics collection.

```bash
# Monitor performance
python scripts/maintenance/performance_monitor.py

# Monitor with custom metrics
python scripts/maintenance/performance_monitor.py --metrics cpu,memory,disk

# Export metrics
python scripts/maintenance/performance_monitor.py --export outputs/metrics.csv
```

### Log Analyzer (`log_analyzer.py`)
Log analysis and reporting with pattern detection and alerting.

```bash
# Analyze logs
python scripts/maintenance/log_analyzer.py

# Analyze specific log file
python scripts/maintenance/log_analyzer.py --log logs/application.log

# Generate analysis report
python scripts/maintenance/log_analyzer.py --report outputs/log_analysis.html
```

### Cleanup System (`cleanup_system.py`)
System cleanup and optimization with configurable policies.

```bash
# Clean system
python scripts/maintenance/cleanup_system.py

# Clean with custom policies
python scripts/maintenance/cleanup_system.py --config config/cleanup.yaml

# Preview cleanup actions
python scripts/maintenance/cleanup_system.py --dry-run
```

## Deployment Scripts (`scripts/deployment/`)

### Build Package (`build_package.py`)
Package building with dependency bundling and optimization.

```bash
# Build package
python scripts/deployment/build_package.py

# Build with specific target
python scripts/deployment/build_package.py --target production

# Build with optimization
python scripts/deployment/build_package.py --optimize
```

### Create Installer (`create_installer.py`)
Installer creation for different platforms with customization options.

```bash
# Create installer
python scripts/deployment/create_installer.py

# Create for specific platform
python scripts/deployment/create_installer.py --platform windows

# Create with custom branding
python scripts/deployment/create_installer.py --branding config/branding.yaml
```

### Deploy Production (`deploy_production.py`)
Production deployment with environment management and validation.

```bash
# Deploy to production
python scripts/deployment/deploy_production.py

# Deploy with validation
python scripts/deployment/deploy_production.py --validate

# Deploy with rollback plan
python scripts/deployment/deploy_production.py --rollback-plan
```

### Update System (`update_system.py`)
System updates with version management and migration support.

```bash
# Update system
python scripts/deployment/update_system.py

# Update to specific version
python scripts/deployment/update_system.py --version 2.1.0

# Update with migration
python scripts/deployment/update_system.py --migrate
```

## Utility Scripts (`scripts/utilities/`)

### Config Manager (`config_manager.py`)
Configuration management with validation and migration support.

```bash
# Validate configuration
python scripts/utilities/config_manager.py --validate

# Migrate configuration
python scripts/utilities/config_manager.py --migrate --from 1.0 --to 2.0

# Export configuration
python scripts/utilities/config_manager.py --export config_backup.yaml
```

### Backup Manager (`backup_manager.py`)
Comprehensive backup management with scheduling and restoration.

```bash
# Create full backup
python scripts/utilities/backup_manager.py --backup

# Restore from backup
python scripts/utilities/backup_manager.py --restore backup_20240115.tar.gz

# Schedule backups
python scripts/utilities/backup_manager.py --schedule daily
```

### Report Generator (`report_generator.py`)
Automated report generation with customizable templates.

```bash
# Generate performance report
python scripts/utilities/report_generator.py --type performance

# Generate custom report
python scripts/utilities/report_generator.py --template templates/custom.html

# Generate and email report
python scripts/utilities/report_generator.py --email admin@company.com
```

### System Info (`system_info.py`)
System information gathering for support and troubleshooting.

```bash
# Gather system info
python scripts/utilities/system_info.py

# Generate support package
python scripts/utilities/system_info.py --support-package

# Export system info
python scripts/utilities/system_info.py --export system_info.json
```

## Script Usage Patterns

### Common Options
Most scripts support these common options:

```bash
--help              # Show help message
--verbose           # Verbose output
--quiet             # Quiet mode (minimal output)
--dry-run           # Preview actions without executing
--config FILE       # Use custom configuration file
--log-level LEVEL   # Set logging level
--output DIR        # Specify output directory
```

### Configuration Files
Scripts can use configuration files for complex operations:

```yaml
# Example: scripts/config/training.yaml
training:
  target_accuracy: 0.95
  max_training_time: 3600
  models:
    - lightgbm
    - xgboost
    - random_forest
  
validation:
  cross_validation_folds: 5
  test_size: 0.2
  
output:
  save_models: true
  generate_report: true
  report_format: html
```

### Scheduling Scripts
Many scripts support scheduling for automation:

```bash
# Add to crontab for daily execution
0 2 * * * /usr/bin/python3 /path/to/scripts/data/collect_data.py

# Windows Task Scheduler
schtasks /create /tn "Gold Trading AI Data Collection" /tr "python scripts/data/collect_data.py" /sc daily /st 02:00
```

### Error Handling
All scripts include comprehensive error handling:

```python
try:
    result = perform_operation()
    logger.info(f"Operation completed successfully: {result}")
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
    sys.exit(1)
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    sys.exit(2)
```

This script collection provides comprehensive automation and management capabilities for the Gold Trading AI system, supporting both development and production environments.
