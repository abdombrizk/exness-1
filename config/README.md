# Configuration Directory

This directory contains all configuration files for the Gold Trading AI system, organized by environment and component.

## Directory Structure

```
config/
├── settings.yaml              # Main application configuration
├── models.yaml               # ML model configurations
├── logging.yaml              # Logging configuration
├── environments/             # Environment-specific configurations
│   ├── development.yaml      # Development environment
│   ├── staging.yaml          # Staging environment
│   └── production.yaml       # Production environment
├── templates/                # Configuration templates
│   ├── settings.template.yaml
│   └── .env.template
└── schemas/                  # Configuration validation schemas
    ├── settings.schema.json
    └── models.schema.json
```

## Configuration Files

### Main Configuration (`settings.yaml`)

The primary configuration file containing all application settings:

```yaml
# Application Settings
application:
  name: "Gold Trading AI"
  version: "2.0.0"
  debug: false
  log_level: "INFO"

# Data Sources
data_sources:
  primary: "yfinance"
  fallback: "synthetic"
  real_time_interval: 30

# Machine Learning
machine_learning:
  target_accuracy: 0.90
  confidence_threshold: 0.60
  ensemble_weights: "auto"

# GUI Configuration
gui:
  theme: "professional_dark"
  window_size: [1600, 1000]
  auto_refresh: true

# Database Configuration
database:
  type: "sqlite"
  path: "data/database/trading_data.db"
  backup_frequency: "daily"
```

### Model Configuration (`models.yaml`)

Specific configurations for ML models:

```yaml
# Model Training Parameters
training:
  train_test_split: 0.8
  validation_split: 0.2
  cross_validation_folds: 5

# Individual Model Settings
models:
  random_forest:
    enabled: true
    n_estimators: 200
    max_depth: 15
    
  xgboost:
    enabled: true
    n_estimators: 300
    learning_rate: 0.1
    
  lightgbm:
    enabled: true
    n_estimators: 300
    num_leaves: 50
```

### Logging Configuration (`logging.yaml`)

Comprehensive logging setup:

```yaml
# Log Levels and Files
logging:
  level: "INFO"
  files:
    application: "logs/application.log"
    trading: "logs/trading.log"
    errors: "logs/errors.log"
    performance: "logs/performance.log"
  
  # Log Rotation
  rotation:
    max_size: "10MB"
    backup_count: 5
```

## Environment-Specific Configurations

### Development (`environments/development.yaml`)

Settings for development environment:

```yaml
# Development overrides
application:
  debug: true
  log_level: "DEBUG"

data_sources:
  primary: "synthetic"  # Use synthetic data for development
  
machine_learning:
  target_accuracy: 0.80  # Lower target for faster development

database:
  path: "data/database/dev_trading_data.db"
```

### Production (`environments/production.yaml`)

Settings for production environment:

```yaml
# Production overrides
application:
  debug: false
  log_level: "WARNING"

data_sources:
  primary: "yfinance"
  backup_enabled: true
  
security:
  encrypt_sensitive_data: true
  rate_limiting: true

performance:
  enable_caching: true
  multiprocessing: true
```

## Configuration Management

### Loading Configuration

The system automatically loads configuration in this order:

1. **Base configuration** from `settings.yaml`
2. **Environment-specific** overrides from `environments/{environment}.yaml`
3. **Environment variables** for sensitive data
4. **Command-line arguments** for runtime overrides

### Environment Variables

Sensitive configuration values should be set via environment variables:

```bash
# API Keys
export ALPHA_VANTAGE_API_KEY="your_key_here"
export QUANDL_API_KEY="your_key_here"

# Database
export DATABASE_URL="sqlite:///data/database/trading_data.db"
export DATABASE_PASSWORD="secure_password"

# Security
export SECRET_KEY="your_secret_key"
export ENCRYPTION_KEY="your_encryption_key"
```

### Using Configuration in Code

```python
from src.utils.config import ConfigManager, get_config

# Load configuration
config = ConfigManager()

# Get specific values
target_accuracy = get_config('machine_learning.target_accuracy', 0.90)
database_path = get_config('database.path')
gui_theme = get_config('gui.theme', 'professional_dark')

# Update configuration
config.set('gui.theme', 'professional_light')
config.save()
```

## Configuration Validation

### Schema Validation

Configuration files are validated against JSON schemas:

```json
{
  "type": "object",
  "properties": {
    "application": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "debug": {"type": "boolean"}
      },
      "required": ["name", "version"]
    }
  }
}
```

### Validation in Code

```python
from src.utils.config import ConfigManager

config = ConfigManager()

# Validate configuration
if config.validate():
    print("Configuration is valid")
else:
    print("Configuration validation failed")
```

## Configuration Templates

### Settings Template (`templates/settings.template.yaml`)

Template for creating new configuration files:

```yaml
# Copy this file to settings.yaml and customize
application:
  name: "Gold Trading AI"
  version: "2.0.0"
  debug: false  # Set to true for development

# Add your API keys here or use environment variables
data_sources:
  alpha_vantage_key: "${ALPHA_VANTAGE_API_KEY}"
  quandl_key: "${QUANDL_API_KEY}"
```

### Environment Template (`.env.template`)

Template for environment variables:

```bash
# Copy this file to .env and add your values

# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
QUANDL_API_KEY=your_quandl_key_here

# Database
DATABASE_URL=sqlite:///data/database/trading_data.db
DATABASE_PASSWORD=your_secure_password

# Security
SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# Environment
ENVIRONMENT=development  # development, staging, production
LOG_LEVEL=INFO
```

## Best Practices

### 1. Security
- Never commit sensitive data (API keys, passwords) to version control
- Use environment variables for sensitive configuration
- Encrypt sensitive data in configuration files

### 2. Environment Separation
- Use different configurations for development, staging, and production
- Keep environment-specific settings in separate files
- Use environment variables to switch between configurations

### 3. Validation
- Validate all configuration values
- Provide sensible defaults for optional settings
- Use type hints and schemas for validation

### 4. Documentation
- Document all configuration options
- Provide examples and templates
- Keep configuration files well-commented

### 5. Versioning
- Version configuration files along with code
- Use migration scripts for configuration changes
- Maintain backward compatibility when possible

## Configuration Reference

### Complete Settings Reference

| Section | Key | Type | Default | Description |
|---------|-----|------|---------|-------------|
| application | name | string | "Gold Trading AI" | Application name |
| application | version | string | "2.0.0" | Application version |
| application | debug | boolean | false | Debug mode flag |
| data_sources | primary | string | "yfinance" | Primary data source |
| data_sources | real_time_interval | integer | 30 | Update interval (seconds) |
| machine_learning | target_accuracy | float | 0.90 | Target model accuracy |
| machine_learning | confidence_threshold | float | 0.60 | Prediction confidence threshold |
| gui | theme | string | "professional_dark" | GUI theme |
| gui | window_size | array | [1600, 1000] | Default window size |
| database | type | string | "sqlite" | Database type |
| database | path | string | "data/database/trading_data.db" | Database file path |

### Model Configuration Reference

| Model | Parameter | Type | Default | Description |
|-------|-----------|------|---------|-------------|
| random_forest | n_estimators | integer | 200 | Number of trees |
| random_forest | max_depth | integer | 15 | Maximum tree depth |
| xgboost | n_estimators | integer | 300 | Number of boosting rounds |
| xgboost | learning_rate | float | 0.1 | Learning rate |
| lightgbm | n_estimators | integer | 300 | Number of boosting rounds |
| lightgbm | num_leaves | integer | 50 | Number of leaves |

This configuration system provides flexibility, security, and maintainability for the Gold Trading AI system across different environments and use cases.
