# Source Code Directory

This directory contains the main source code for the Gold Trading AI system, organized following enterprise software engineering best practices.

## Directory Structure

```
src/
├── core/                    # Core business logic and components
│   ├── models/             # Machine learning models and training
│   ├── data/               # Data management and processing
│   ├── database/           # Database operations and management
│   ├── analysis/           # Technical and fundamental analysis
│   └── risk/               # Risk management and position sizing
├── gui/                    # Graphical user interface components
│   ├── application.py      # Main application window
│   ├── components/         # Reusable UI components
│   ├── themes/             # Professional themes and styling
│   ├── charts/             # Chart components and visualizations
│   └── dialogs/            # Dialog windows and forms
└── utils/                  # Utility functions and helpers
    ├── config.py           # Configuration management
    ├── logger.py           # Logging system
    ├── validators.py       # Data validation and sanitization
    ├── helpers.py          # Common helper functions
    └── constants.py        # System constants and enumerations
```

## Core Modules

### 🤖 Core (`src/core/`)
Contains the main business logic and core functionality:

- **Models**: ML training, prediction, and model management
- **Data**: Data fetching, processing, and feature engineering
- **Database**: Data storage, retrieval, and management
- **Analysis**: Technical and fundamental analysis engines
- **Risk**: Risk management and position sizing algorithms

### 🖥️ GUI (`src/gui/`)
Professional Bloomberg Terminal-style user interface:

- **Application**: Main application window and controller
- **Components**: Reusable UI widgets and components
- **Themes**: Professional dark/light themes
- **Charts**: Interactive price charts and visualizations
- **Dialogs**: Settings, about, and configuration dialogs

### 🔧 Utils (`src/utils/`)
Common utilities and helper functions:

- **Config**: Centralized configuration management
- **Logger**: Professional logging system with multiple handlers
- **Validators**: Data validation and sanitization
- **Helpers**: Common utility functions
- **Constants**: System-wide constants and enumerations

## Import Structure

The source code follows a hierarchical import structure:

```python
# Top-level imports
from src import ModelManager, DataManager, GoldTradingApp

# Core module imports
from src.core.models import AdvancedMLTrainer, ModelPredictor
from src.core.data import DataProcessor, FeatureEngineer
from src.core.database import DatabaseManager, TradingDatabase

# GUI imports
from src.gui.application import GoldTradingApp
from src.gui.components import ProfessionalTheme, ChartWidget

# Utility imports
from src.utils.config import ConfigManager, get_config
from src.utils.logger import get_logger, log_performance
```

## Design Principles

### 1. Separation of Concerns
Each module has a single, well-defined responsibility:
- Core modules handle business logic
- GUI modules handle user interface
- Utils modules provide common functionality

### 2. Dependency Injection
Components accept dependencies through their constructors, making them testable and flexible.

### 3. Configuration-Driven
All components are configurable through the centralized configuration system.

### 4. Professional Logging
Comprehensive logging throughout the system with structured log formats.

### 5. Error Handling
Robust error handling with graceful degradation and user-friendly error messages.

## Code Quality Standards

### Type Hints
All functions and methods include type hints for better code documentation and IDE support:

```python
def process_data(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Process market data with specified configuration."""
    pass
```

### Documentation
All modules, classes, and functions include comprehensive docstrings:

```python
class ModelManager:
    """
    Centralized model management system.
    
    Handles model loading, saving, versioning, and performance tracking.
    Provides a unified interface for all ML model operations.
    """
    pass
```

### Error Handling
Consistent error handling patterns throughout the codebase:

```python
try:
    result = risky_operation()
    logger.info("Operation completed successfully")
    return result
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
    raise
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise
```

## Testing

Each source module has corresponding tests in the `tests/` directory:

```
tests/
├── unit/
│   ├── core/
│   ├── gui/
│   └── utils/
├── integration/
└── performance/
```

Run tests for specific modules:

```bash
# Test core modules
python -m pytest tests/unit/core/ -v

# Test GUI components
python -m pytest tests/unit/gui/ -v

# Test utilities
python -m pytest tests/unit/utils/ -v
```

## Development Guidelines

### Adding New Features

1. **Create the module** in the appropriate directory
2. **Add comprehensive tests** in the corresponding test directory
3. **Update the `__init__.py`** files to expose the new functionality
4. **Add configuration options** if needed
5. **Update documentation** including this README

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Keep functions focused and small
- Add type hints and docstrings
- Handle errors appropriately

### Performance Considerations

- Use efficient algorithms and data structures
- Cache expensive operations when appropriate
- Profile code for performance bottlenecks
- Use vectorized operations for data processing

## Integration Points

### Configuration
All modules use the centralized configuration system:

```python
from src.utils.config import get_config

# Get configuration values
target_accuracy = get_config('machine_learning.target_accuracy', 0.90)
database_path = get_config('database.path', 'data/database/trading_data.db')
```

### Logging
All modules use the centralized logging system:

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Module initialized successfully")
```

### Database
All modules use the centralized database manager:

```python
from src.core.database import DatabaseManager

db_manager = DatabaseManager()
predictions = db_manager.get_predictions()
```

This structure ensures maintainability, testability, and scalability of the Gold Trading AI system.
