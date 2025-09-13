# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance Chinese stock technical analysis system focused on buy-point analysis and technical indicator calculations. The system achieves 99.9% performance optimization, processing stocks at 0.05 seconds per stock with 72,000 stocks/hour throughput capability.

### Key Features
- **88+ Technical Indicators**: Complete coverage of trend, oscillation, volume, volatility indicators
- **ZXM Professional System**: 25+ specialized ZXM indicators for buy-point detection
- **Vectorized Computing**: 40-70% performance improvement using numpy/pandas
- **Intelligent Caching**: LRU memory cache + disk persistence with 50% hit rate
- **Parallel Processing**: 8-process parallel execution with 800% CPU utilization improvement

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov

# Run specific test modules
python -m pytest tests/test_indicators.py
python -m pytest tests/test_api.py
```

### Code Quality and Linting
```bash
# Format code with Black (line length: 120)
black analysis/ indicators/ utils/ db/ --line-length 120

# Check code style with flake8
flake8 analysis/ indicators/ utils/ db/

# Sort imports with isort
isort analysis/ indicators/ utils/ db/ --profile black
```

### Running the System

#### Main Analysis Entry Points
```bash
# Main system entry point
python bin/main.py

# Stock selection
python bin/stock_select.py

# Parallel buy-point analysis
python bin/buypoint_batch_analyzer.py

# Performance testing
python bin/quick_performance_test.py
python bin/simple_performance_test.py
```

#### API Server
```bash
# Start FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python directly
python api/main.py
```

## Architecture Overview

### Core Components

**Database Layer (`db/`)**
- **DataAccessManager**: Unified data access interface with caching and connection management
- **ClickHouse Integration**: Primary database for stock data storage and retrieval
- **Connection Pooling**: Optimized connection management for high-performance queries
- **Multi-layer Caching**: Memory + disk caching with intelligent cache management

**Indicators System (`indicators/`)**
- **CompleteIndicatorRegistry**: Registry for all 88+ technical indicators
- **Core Indicators**: RSI, MACD, KDJ, BOLL, ATR and other fundamental indicators
- **ZXM Professional System**: Specialized trend detection and buy-point analysis indicators
- **Pattern Recognition**: K-line pattern and price pattern detection indicators
- **Enhanced Indicators**: Optimized versions with improved performance and accuracy

**Analysis Engine (`analysis/`)**
- **Buy-point Analysis**: Core buy-point detection and scoring algorithms
- **Market Analysis**: Market-wide analysis and indicator calculations
- **Performance Optimization**: Vectorized computations and parallel processing
- **Intelligent Caching**: Result caching system for performance optimization

**Strategy System (`strategy/`)**
- **BaseStrategy**: Abstract base class for all trading strategies
- **Strategy Factory**: Dynamic strategy creation and management
- **Backtesting**: Strategy performance evaluation and historical testing

**API Layer (`api/`)**
- **RESTful API**: Complete REST API for stock data and analysis
- **Real-time Monitoring**: Live market monitoring and alerting
- **Risk Management**: Risk monitoring and alert system

### Key Design Patterns

1. **Dependency Injection**: Uses container-based dependency injection for service management
2. **Factory Pattern**: StrategyFactory and IndicatorFactory for dynamic object creation
3. **Singleton Pattern**: Database connection management and configuration
4. **Repository Pattern**: Data access abstraction through DataAccessInterface
5. **Observer Pattern**: Event-driven monitoring and alerting system

### Data Flow

```
Stock Data Input → DataAccessManager → Caching Layer → Analysis Engine → Results Output
                                  ↗ ClickHouse DB ↗
```

### Performance Architecture

```
Parallel Processing (8 workers) → Vectorized Computing → Intelligent Caching → Output
```

## Configuration

### Database Configuration
- Primary database: ClickHouse
- Connection managed through `config/database.yml`
- Database credentials via environment variables or encrypted config

### System Configuration
- Main config in `config/` directory
- Environment-specific overrides supported
- Sensitive information encrypted or in environment variables

## Common Development Tasks

### Adding New Indicators
1. Inherit from `BaseIndicator` in `indicators/core/`
2. Implement required abstract methods
3. Register in `CompleteIndicatorRegistry`
4. Add tests in `tests/test_indicators.py`

### Creating New Strategies
1. Inherit from `BaseStrategy` in `strategy/`
2. Implement `select()` method
3. Register in `StrategyFactory`
4. Add backtesting configuration

### Working with Database
- Use `DataAccessManager` for all database operations
- Leverage caching for performance
- Follow the repository pattern for data access

### Performance Optimization
- Use vectorized operations for bulk calculations
- Implement caching for expensive computations
- Consider parallel processing for independent operations
- Monitor performance with built-in profiling tools

## File Structure Conventions

### Directory Organization
- `bin/`: Executable scripts and main entry points
- `analysis/`: Core analysis algorithms and engines
- `indicators/`: Technical indicator implementations
- `strategy/`: Trading strategy implementations
- `db/`: Database access layer and managers
- `utils/`: Utility functions and common tools
- `config/`: Configuration files and management
- `tests/`: Test suites and test utilities
- `api/`: REST API implementations
- `monitoring/`: System monitoring and alerting
- `docs/`: Documentation and guides

### Naming Conventions
- Classes: PascalCase (e.g., `DataAccessManager`)
- Functions/Methods: snake_case (e.g., `get_stock_data`)
- Variables: snake_case (e.g., `stock_code`)
- Constants: UPPER_SNAKE_CASE (e.g., `GLOBAL_DATE`)
- Files: snake_case (e.g., `data_access_manager.py`)

### Import Standards
- Use absolute imports from project root
- Import order: standard library → third-party → project modules
- Group related imports and separate with blank lines

## Testing Strategy

### Test Organization
- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for component interaction
- `tests/api/`: API endpoint testing
- Use pytest framework with coverage reporting

### Test Data
- Use ClickHouse database for real data testing
- Mock external dependencies in unit tests
- Maintain test data integrity across test runs

## API Development

### FastAPI Application
- Main app in `api/main.py`
- Router organization in `api/routers/`
- Automatic OpenAPI documentation
- CORS enabled for web client integration

### Monitoring and Alerting
- Real-time market monitoring in `monitoring/`
- Intelligent alert system with configurable rules
- Risk monitoring and automated alerts
- Performance monitoring with system metrics

## Special Notes

### Project Rules (.trae/rules/project_rules.md)
- Follow established coding standards and architecture patterns
- Use dependency injection for service management
- Implement proper exception handling and logging
- Maintain configuration management best practices
- Follow the established project workflow (requirements → design → implementation → testing)

### Performance Considerations
- This system is optimized for high-performance stock analysis
- Always consider caching for expensive operations
- Use vectorized operations when possible
- Leverage parallel processing for independent tasks
- Monitor performance metrics regularly

### Database Dependencies
- All data operations use ClickHouse as the primary database
- No simulated data - real market data only
- Connection pooling and caching are critical for performance
- Follow the unified data access patterns