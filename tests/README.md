# Test Suite Directory

This directory contains the comprehensive test suite for the Gold Trading AI system, organized by test type and component.

## Directory Structure

```
tests/
├── unit/                       # Unit tests for individual components
│   ├── core/                  # Core module tests
│   │   ├── test_models.py     # ML model tests
│   │   ├── test_data.py       # Data management tests
│   │   ├── test_database.py   # Database tests
│   │   ├── test_analysis.py   # Analysis engine tests
│   │   └── test_risk.py       # Risk management tests
│   ├── gui/                   # GUI component tests
│   │   ├── test_application.py # Main application tests
│   │   ├── test_components.py  # UI component tests
│   │   └── test_themes.py      # Theme tests
│   └── utils/                 # Utility tests
│       ├── test_config.py     # Configuration tests
│       ├── test_logger.py     # Logging tests
│       └── test_validators.py # Validation tests
├── integration/               # Integration tests
│   ├── test_ml_pipeline.py   # ML training pipeline
│   ├── test_data_flow.py     # End-to-end data flow
│   ├── test_gui_backend.py   # GUI-backend integration
│   └── test_database_ops.py  # Database operations
├── performance/               # Performance and benchmark tests
│   ├── test_model_speed.py   # Model training/prediction speed
│   ├── test_data_processing.py # Data processing performance
│   ├── test_memory_usage.py  # Memory usage tests
│   └── benchmarks.py         # Performance benchmarks
├── fixtures/                  # Test data and fixtures
│   ├── sample_data/          # Sample market data
│   ├── mock_responses/       # Mock API responses
│   └── test_configs/         # Test configurations
├── conftest.py               # Pytest configuration and fixtures
├── run_all_tests.py          # Test runner script
└── validation_suite.py       # Production readiness validation
```

## Test Categories

### 🧪 Unit Tests (`tests/unit/`)

Test individual components in isolation:

**Core Module Tests:**
- Model training and prediction accuracy
- Data fetching and processing
- Database CRUD operations
- Technical analysis calculations
- Risk management algorithms

**GUI Component Tests:**
- Widget functionality
- Theme application
- Chart rendering
- User interaction handling

**Utility Tests:**
- Configuration loading and validation
- Logging functionality
- Data validation
- Helper functions

### 🔗 Integration Tests (`tests/integration/`)

Test component interactions and workflows:

**ML Pipeline Integration:**
- Data → Feature Engineering → Training → Prediction
- Model saving and loading
- Performance tracking

**Data Flow Integration:**
- Real-time data → Processing → Storage → Display
- Error handling and recovery
- Data consistency

**GUI-Backend Integration:**
- User actions → Backend processing → UI updates
- Real-time updates
- Configuration changes

### ⚡ Performance Tests (`tests/performance/`)

Test system performance and benchmarks:

**Speed Tests:**
- Model training time
- Prediction latency
- Data processing throughput
- GUI responsiveness

**Resource Usage:**
- Memory consumption
- CPU utilization
- Disk I/O
- Network usage

**Scalability Tests:**
- Large dataset handling
- Concurrent operations
- Long-running processes

## Running Tests

### Run All Tests
```bash
# Complete test suite
python tests/run_all_tests.py

# Using pytest directly
pytest tests/ -v --cov=src

# Parallel execution
pytest tests/ -n auto
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Performance tests only
pytest tests/performance/ -v

# Specific component tests
pytest tests/unit/core/test_models.py -v
```

### Run Tests with Coverage
```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Run Performance Benchmarks
```bash
# Run benchmarks
python tests/performance/benchmarks.py

# Compare with previous results
pytest tests/performance/ --benchmark-compare
```

## Test Configuration

### Pytest Configuration (`conftest.py`)

```python
import pytest
import tempfile
from pathlib import Path
from src.utils.config import ConfigManager

@pytest.fixture
def temp_config():
    """Temporary configuration for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.yaml"
        config = ConfigManager(config_path)
        yield config

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
        'open': np.random.normal(2000, 50, 100),
        'high': np.random.normal(2010, 50, 100),
        'low': np.random.normal(1990, 50, 100),
        'close': np.random.normal(2000, 50, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })
```

### Test Markers

Tests are organized using pytest markers:

```python
@pytest.mark.unit
def test_model_training():
    """Unit test for model training"""
    pass

@pytest.mark.integration
def test_data_pipeline():
    """Integration test for data pipeline"""
    pass

@pytest.mark.performance
def test_prediction_speed():
    """Performance test for prediction speed"""
    pass

@pytest.mark.slow
def test_full_training():
    """Slow test that takes significant time"""
    pass
```

### Running Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run only fast tests
pytest -m "not slow"

# Run integration and performance tests
pytest -m "integration or performance"
```

## Test Data and Fixtures

### Sample Data (`tests/fixtures/sample_data/`)

**Market Data:**
- `gold_prices_sample.csv` - Sample gold price data
- `economic_indicators_sample.csv` - Sample economic data
- `news_sentiment_sample.json` - Sample news sentiment data

**Model Data:**
- `trained_model_sample.joblib` - Sample trained model
- `feature_data_sample.csv` - Sample feature data
- `prediction_results_sample.csv` - Sample prediction results

### Mock Responses (`tests/fixtures/mock_responses/`)

**API Responses:**
- `yfinance_response.json` - Mock Yahoo Finance response
- `alpha_vantage_response.json` - Mock Alpha Vantage response
- `news_api_response.json` - Mock news API response

### Test Configurations (`tests/fixtures/test_configs/`)

**Configuration Files:**
- `test_settings.yaml` - Test configuration
- `minimal_config.yaml` - Minimal configuration for testing
- `performance_config.yaml` - Configuration for performance tests

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest tests/unit/ -x
        language: system
        pass_filenames: false
        always_run: true
```

## Test Quality Standards

### Code Coverage
- **Minimum Coverage**: 95%
- **Critical Components**: 100% coverage required
- **Exclusions**: Only for unreachable code (error handling)

### Test Naming Convention
```python
def test_[component]_[action]_[expected_result]():
    """Test that [component] [action] results in [expected_result]"""
    pass

# Examples:
def test_model_training_with_valid_data_succeeds():
def test_database_connection_with_invalid_path_raises_error():
def test_gui_theme_change_updates_colors():
```

### Test Structure (AAA Pattern)
```python
def test_example():
    # Arrange - Set up test data and conditions
    data = create_test_data()
    model = ModelTrainer()
    
    # Act - Execute the code being tested
    result = model.train(data)
    
    # Assert - Verify the results
    assert result.accuracy > 0.9
    assert result.model is not None
```

## Debugging Tests

### Running Tests in Debug Mode
```bash
# Run with debugging
pytest tests/unit/test_models.py -s -vv --pdb

# Run specific test with debugging
pytest tests/unit/test_models.py::test_model_training -s --pdb
```

### Test Logging
```python
import logging

def test_with_logging(caplog):
    """Test with log capture"""
    with caplog.at_level(logging.INFO):
        result = function_under_test()
        
    assert "Expected log message" in caplog.text
```

This comprehensive test suite ensures the reliability, performance, and quality of the Gold Trading AI system across all components and use cases.
