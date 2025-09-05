# AI Trading Advisor

AI-powered trading advisor with dynamic risk management, multi-source data integration, and automated trading recommendations.

## Features

- 🎯 **Dynamic Risk Management** - Personalized risk profiles with real-time adjustments
- 📊 **Multi-Source Data Integration** - Fundamentals, technicals, news, and sentiment analysis
- 🤖 **AI-Powered Recommendations** - Daily buy/sell signals with decision transparency
- ⚡ **Multi-Timeframe Analysis** - Intraday, short-term, ETF, and long-term strategies
- 🔄 **Automated Pipeline** - Apache Airflow orchestration for reliable data processing
- 🛡️ **Enterprise Security** - Comprehensive risk controls and audit logging

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-trading-advisor

# Initial setup (optional - for local development)
make setup

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and settings
```

### **🚀 Fastest Setup (Recommended)**

Start Docker first, then run our automated setup:

```bash
# Start Docker service (choose one that works for your system)
sudo service docker start
# OR sudo systemctl start docker
# OR start Docker Desktop

# Run complete setup with sample data
make trigger-and-wait
```

This single command will:
- ✅ Build all Docker containers
- ✅ Start PostgreSQL, Redis, and all services
- ✅ Create database schema with all required tables
- ✅ Populate sample trading data (market data, news, analysis, recommendations)
- ✅ Verify everything is working

### **🧪 Testing Setup**

```bash
# Build test container with pytest dependencies
make docker-test-build

# Run all tests in Docker container
make docker-test
```

### Quick Verification

```bash
# Check database has sample data
make db-show-data

# Check API health
curl http://localhost:8000/health

# Check Airflow UI (optional)
make airflow-init && make airflow-start
open http://localhost:8080  # admin/admin
```

### Testing Phase 1 (Infrastructure)

#### 1. **Basic Configuration Test**
```bash
# Run configuration and unit tests
make test

# Expected output: All tests pass with risk management validation
# ✅ Test results: 19 passed, 0 failed
# ✅ Risk management calculations validated
# ✅ Configuration loading successful
```

#### 2. **API Health Check**
```bash
# Start development server
make dev

# Test endpoints (in another terminal)
curl http://localhost:8000/
curl http://localhost:8000/health

# Expected responses:
# {"message":"AI Trading Advisor API","version":"0.1.0","status":"running"}
# {"status":"healthy","timestamp":"2024-01-01T00:00:00Z"}
```

#### 3. **Docker Services Test**
```bash
# Build and start all services
make docker-build
make docker-up

# Check running services
docker ps

# Expected: PostgreSQL, Redis, and Airflow containers running
# Test database connection
docker exec ai-trading-advisor-postgres-1 pg_isready -U trader
```

#### 4. **Database Schema Verification**
```bash
# Start database services
make db-up

# Run database migrations
make db-migrate

# Verify database schema
docker exec -it ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "\dt"

# Expected tables:
# - users, portfolios, positions
# - market_data, news_sentiment
# - recommendations, risk_assessments
# - analysis_results, trading_signals
```

#### 5. **Airflow DAG Validation**
```bash
# Initialize Airflow database
make airflow-init

# If initialization fails, use manual method:
make airflow-init-manual

# Start Airflow services
make airflow-start

# Check database status
make airflow-db-check

# Validate DAG syntax and list DAGs
make airflow-dags-list

# Expected DAGs:
# - data_collection_pipeline (every 15 minutes)
# - analysis_pipeline (every 30 minutes)
# - recommendation_pipeline (every hour)

# Access Airflow UI: http://localhost:8080 (admin/admin)
```

#### 6. **Pipeline Integration Test**
```bash
# Test data collection pipeline
make test-data-pipeline

# Test analysis pipeline  
make test-analysis-pipeline

# Test recommendation pipeline
make test-recommendation-pipeline

# Expected: Each pipeline completes without errors
# Check logs: make airflow-logs
```

#### 7. **Risk Management Validation**
```bash
# Test position size calculations 
```
docker exec -it ai-trading-advisor-api-1 \
python -c "from src.config import RiskConfig; result = RiskConfig.calculate_position_size(10000, 0.02, 0.05); print(f'Position size: ${result}'); assert result == 1000.0, 'Risk calculation failed'; print('✅ Risk management validation passed')"
```

# Test risk limit enforcement
make test-risk-limits

# Expected: All risk calculations respect configured limits
```

#### 8. **Code Quality Verification**
```bash
# Format code
make format

# Run linting
make lint #todo: to fix

# Run tests with coverage
make test-coverage

# Expected results:
# ✅ Code formatted with black and isort
# ✅ No linting errors (flake8, mypy)
# ✅ Test coverage > 80%
# ✅ All type hints validated
```

#### 9. **Environment Configuration Test**
```bash
# Validate environment variables
python -c "
from src.config import settings
print('Database URL:', settings.database_url[:20] + '...')
print('Redis URL:', settings.redis_url)
print('Max risk per trade:', settings.max_risk_per_trade)
print('✅ Environment configuration loaded')
"

# Test API key validation (if keys provided)
make test-api-keys

# Expected: Configuration loaded without errors
```

#### 10. **End-to-End Health Check**
```bash
# Run comprehensive health check
make health-check

# This command tests:
# - API responsiveness
# - Database connectivity  
# - Redis connectivity
# - Airflow scheduler status
# - All services integration

# Expected: All systems healthy
```

### Troubleshooting

#### **🔥 Quick Fixes for Common Issues**

**Docker not running:**
```bash
# Start Docker service (choose one that works)
sudo service docker start
sudo systemctl start docker
# OR start Docker Desktop manually

# Then run setup
make trigger-and-wait
```

**Database has no data / "zero data" error:**
```bash
# This is automatically fixed by:
make trigger-and-wait

# Verify data was populated:
make db-show-data
```

**pytest not found in Docker:**
```bash
# Build test container first:
make docker-test-build

# Then run tests:
make docker-test
```

#### **🐛 Advanced Troubleshooting**

**Port 8000 already in use:**
```bash
# Kill existing processes
pkill -f "uvicorn.*main:app"
# Or use different port
make dev PORT=8001
```

**Database connection failed:**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres
# Restart database
make db-reset
```

**Airflow DAGs not showing:**
```bash
# Check DAG directory mount
docker exec ai-trading-advisor-airflow-webserver-1 ls /opt/airflow/dags

# Check Airflow database status
make airflow-db-check

# List DAGs
make airflow-dags-list

# View logs for errors
make airflow-logs
```

**Test failures:**
```bash
# Run specific test in Docker
make docker-test-specific FILE=tests/test_database.py

# Interactive debugging
make docker-test-interactive

# Local testing (if setup)
pytest tests/test_config.py::TestRiskConfig::test_calculate_position_size_normal -v
```

**Container build failures:**
```bash
# Clean rebuild everything
docker-compose down -v
docker system prune -f
make trigger-and-wait
```

### Service Endpoints

- **API**: http://localhost:8000
- **Airflow**: http://localhost:8080 (admin/admin)
- **PostgreSQL**: localhost:5432 (trader/trader_password)
- **Redis**: localhost:6379

### Development Commands

#### **🚀 Essential Commands**

```bash
# Complete setup with sample data
make trigger-and-wait          # Start everything + populate database

# Testing
make docker-test-build         # Build test container with pytest
make docker-test              # Run all tests in Docker
make docker-test-coverage     # Run tests with coverage report
make docker-test-interactive  # Interactive test container for debugging

# Database verification
make db-show-data             # Show sample data from all tables
make db-show-tables           # List all database tables
```

#### **🛠️ Development & Debugging**

```bash
# Local development
make setup                    # Initial project setup (Python venv)
make dev                      # Start development server locally
make test                     # Run test suite locally
make test-coverage            # Run tests with coverage locally
make lint                     # Run code linting
make format                   # Format code

# Docker Operations
make docker-build             # Build Docker containers
make docker-up                # Start all services
make docker-down              # Stop all services
make docker-restart           # Restart all services

# Specific testing
make docker-test-api          # Test FastAPI endpoints
make docker-test-data         # Test database operations
make docker-test-specific FILE=tests/test_example.py  # Test specific file
```

#### **⚙️ Advanced Operations**

```bash
# Airflow Operations
make airflow-init             # Initialize Airflow database
make airflow-init-manual      # Manual Airflow initialization
make airflow-start            # Start Airflow services
make airflow-db-check         # Check Airflow database status
make airflow-dags-list        # List all DAGs
make airflow-logs             # View Airflow logs

# Database Operations
make db-up                    # Start database services only
make db-reset                 # Reset database
make db-connect               # Interactive PostgreSQL connection

# Health & Monitoring
make health                   # Check service health
make health-check             # Comprehensive system health check
make clean                    # Clean temporary files
```

## Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Apache        │    │   PostgreSQL    │
│   REST API      │◄──►│   Airflow       │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Risk Engine   │    │   Data Pipeline │    │   Redis Cache   │
│   & Analytics   │    │   & Processing  │    │   & Queue       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Airflow DAG Architecture
```
┌────────────────────────────────────────────────────────────────┐
│                    Data Collection Pipeline                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Market Data  │  │ News & Sent  │  │ Data Quality & Store │  │
│  │ (15min)      │→ │ (15min)      │→ │ (15min)              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────┐
│                     Analysis Pipeline                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Technical    │  │ Fundamental  │  │ Risk & Sentiment     │  │
│  │ Analysis     │  │ Analysis     │  │ Analysis             │  │
│  │ (30min)      │  │ (30min)      │  │ (30min)              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────┐
│                  Recommendation Pipeline                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Signal Gen   │  │ Position     │  │ Risk Filter & Report │  │
│  │ (1hour)      │→ │ Sizing       │→ │ (1hour)              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Docker Components
- **Dockerfile.airflow**: Custom Airflow image with dependencies from `requirements-airflow.txt`
- **Dockerfile**: FastAPI application with main dependencies
- **Multi-stage builds**: Separated FastAPI and Airflow dependencies for cleaner builds

## Project Status

- ✅ **Phase 1 Complete**: Infrastructure setup, Docker, Airflow, database schema
- 🚧 **Phase 2 In Progress**: Data collection engines, risk management
- ⏳ **Phase 3 Planned**: Analysis engines, recommendation system
- ⏳ **Phase 4 Planned**: Trading integration, performance tracking

## Contributing

1. Follow PEP 8 coding standards
2. Add tests for new features
3. Update documentation
4. Run `make lint` and `make test` before committing

## License

MIT License - see LICENSE file for details

## Project structure
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.airflow
├── init-db-simple.sql
├── init-db.sql
├── Makefile
├── populate_data.sql
├── populate_sample_data.py
├── pyproject.toml
├── README.md
├── requirements-airflow.txt
├── requirements-dev.txt
├── requirements.txt
├── src
│   ├── airflow_dags
│   │   ├── analysis_pipeline.py
│   │   ├── data_pipeline.py
│   │   ├── __init__.py
│   │   ├── recommendation_pipeline.py
│   │   └── simple_data_pipeline.py
│   ├── api
│   │   └── __init__.py
│   ├── config.py
│   ├── core
│   │   ├── analysis_engine.py
│   │   ├── __init__.py
│   │   ├── recommendation_engine.py
│   │   └── risk_engine.py
│   ├── data
│   │   ├── collectors.py
│   │   ├── database.py
│   │   ├── __init__.py
│   │   └── processors.py
│   ├── __init__.py
│   └── main.py
└── tests
    ├── __init__.py
    ├── test_config.py
    ├── test_database.py
    └── test_main.py
