.PHONY: setup install-deps create-env dev test lint format clean docker-build docker-up docker-down

# Docker Compose command (v2 or fallback)
DOCKER_COMPOSE := $(shell (docker compose version >/dev/null 2>&1 && echo "docker compose") || echo "docker-compose")

# Variables
PYTHON := python3.11
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Setup and Installation
setup: install-deps create-env
	@echo "Setup complete! Run 'make dev' to start development server"

install-deps:
	@echo "Checking Python venv support..."
	@if ! $(PYTHON) -m venv --help >/dev/null 2>&1; then \
		echo "Python venv module missing. Installing required system packages..."; \
		sudo apt update && sudo apt install -y python3.11-venv python3.11-distutils; \
	fi
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Upgrading pip in virtual environment..."
	$(PYTHON_VENV) -m ensurepip --upgrade
	$(PYTHON_VENV) -m pip install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt

create-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from .env.example"; \
		echo "Please update the .env file with your actual API keys and settings"; \
	fi

# Development
dev:
	$(PYTHON_VENV) -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Code Quality
format:
	$(VENV)/bin/black src tests
	$(VENV)/bin/isort src tests

lint:
	$(VENV)/bin/flake8 src tests
	$(VENV)/bin/mypy src

# Testing
test:
	$(VENV)/bin/pytest tests/test_core_functionality.py::TestConfigurationSettings tests/test_core_functionality.py::TestDatabaseSchema tests/test_core_functionality.py::TestDataValidationLogic tests/test_core_functionality.py::TestDataFormats tests/test_core_functionality.py::TestEnvironmentConfiguration tests/test_core_functionality.py::TestIntegrationReadiness tests/test_data_collectors.py::TestMarketDataCollector::test_init tests/test_data_collectors.py::TestMarketDataCollector::test_collect_real_time_data_dummy_mode tests/test_data_collectors.py::TestNewsCollector::test_collect_financial_news_dummy_mode tests/test_strategic_collectors.py::TestFundamentalDataCollector::test_init tests/test_strategic_collectors.py::TestFundamentalDataCollector::test_collect_weekly_fundamentals_dummy_mode tests/test_strategic_collectors.py::TestVolatilityMonitor::test_init tests/test_strategic_collectors.py::TestVolatilityMonitor::test_check_market_volatility_dummy_mode -v

test-all:
	@echo "Running all tests for simplified architecture (4 test files)..."
	@echo "Testing: data_manager, analysis_engine, trading_engine, and all DAGs..."
	@POSTGRES_HOST=localhost POSTGRES_DB=airflow POSTGRES_USER=airflow POSTGRES_PASSWORD=airflow \
	$(VENV)/bin/pytest tests/test_data_manager.py tests/test_analysis_engine.py tests/test_trading_engine.py tests/test_data_collection_dag.py tests/test_analysis_dag.py tests/test_trading_dag.py \
	-v --tb=short

test-dags:
	@echo "Running DAG-specific tests..."
	@POSTGRES_HOST=localhost POSTGRES_DB=airflow POSTGRES_USER=airflow POSTGRES_PASSWORD=airflow \
	$(VENV)/bin/pytest tests/test_data_collection_dag.py tests/test_analysis_dag.py tests/test_trading_dag.py -v

test-core:
	@echo "Running core functionality tests for simplified architecture..."
	@POSTGRES_HOST=localhost POSTGRES_DB=airflow POSTGRES_USER=airflow POSTGRES_PASSWORD=airflow \
	$(VENV)/bin/pytest tests/test_data_manager.py tests/test_analysis_engine.py tests/test_trading_engine.py \
	--tb=short -q

test-coverage:
	$(VENV)/bin/pytest tests/ --cov=src --cov-report=html --cov-report=term

test-watch:
	$(VENV)/bin/pytest tests/ -v --tb=short -f

# Docker Testing Commands
docker-test-build:
	@echo "Building test container with dependencies..."
	$(DOCKER_COMPOSE) build test

docker-test:
	@echo "Running tests in Docker container..."
	$(DOCKER_COMPOSE) run --rm test python -m pytest tests/ -v

docker-test-coverage:
	@echo "Running tests with coverage in Docker container..."
	$(DOCKER_COMPOSE) run --rm test python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

docker-test-watch:
	@echo "Running tests in watch mode in Docker container..."
	$(DOCKER_COMPOSE) run --rm test python -m pytest tests/ -v --tb=short -f

docker-test-specific:
	@echo "Running specific test file in Docker container..."
	@echo "Usage: make docker-test-specific FILE=tests/test_example.py"
	$(DOCKER_COMPOSE) run --rm test python -m pytest $(FILE) -v

docker-test-interactive:
	@echo "Starting interactive test container..."
	$(DOCKER_COMPOSE) run --rm test bash

# Quick test commands for specific areas
docker-test-api:
	$(DOCKER_COMPOSE) run --rm test python -m pytest tests/test_main.py -v

docker-test-engines:
	$(DOCKER_COMPOSE) run --rm test python -m pytest tests/test_engines.py -v

docker-test-data:
	$(DOCKER_COMPOSE) run --rm test python -m pytest tests/test_database.py -v

# Docker Commands
docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up -d

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

docker-restart:
	$(DOCKER_COMPOSE) restart

# Airflow Commands
airflow-init:
	@echo "🚀 Initializing Airflow (Compatible with Docker Compose v2.35.1)"
	@echo "Step 1: Starting Airflow database..."
	@$(DOCKER_COMPOSE) up -d airflow-postgres
	@echo "Step 2: Building and starting Airflow services..."
	@$(DOCKER_COMPOSE) build airflow-webserver airflow-scheduler
	@$(DOCKER_COMPOSE) up -d airflow-webserver airflow-scheduler
	@echo "✅ Airflow services started!"
	@echo ""
	@echo "🔧 MANUAL STEP REQUIRED due to Docker Compose v2.35.1 compatibility issues:"
	@echo "   Wait 30-60 seconds for Airflow to fully initialize, then run:"
	@echo "   docker exec ai-trading-advisor-airflow-webserver-1 airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin"
	@echo "   OR try: make airflow-create-user"
	@echo ""
	@echo "🌐 Airflow webserver will be available at: http://localhost:8080"
	@echo "👤 Default login: admin / admin"
	@echo ""
	@echo "📊 Check status with: make airflow-status"

airflow-init-manual:
	@echo "Using alternative Airflow initialization due to Docker Compose issues..."
	@echo "Building Airflow containers first..."
	$(DOCKER_COMPOSE) build airflow-init airflow-webserver airflow-scheduler
	@echo "Creating airflow user in database directly..."
	docker run --rm --network ai-trading-advisor_default -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow_password@airflow-postgres:5432/airflow ai-trading-advisor-airflow-init bash -c "airflow db init && airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin || echo 'Init completed'"

airflow-start:
	$(DOCKER_COMPOSE) up -d airflow-webserver airflow-scheduler

airflow-create-user:
	@echo "Creating Airflow admin user..."
	@docker exec ai-trading-advisor-airflow-webserver-1 airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin 2>/dev/null || echo "User already exists or Airflow not ready"
	@echo "✅ Admin user setup complete! Login: admin / admin"

airflow-logs:
	$(DOCKER_COMPOSE) logs -f airflow-webserver airflow-scheduler

airflow-db-check:
	@if docker ps | grep -q airflow; then \
		docker exec -u airflow ai-trading-advisor_airflow_1 airflow db check; \
	else \
		echo "Airflow container is not running. Start it with 'make airflow-start'"; \
	fi

airflow-dags-list:
	@if docker ps | grep -q airflow; then \
		docker exec ai-trading-advisor_airflow-webserver_1 airflow dags list; \
	else \
		echo "Airflow container is not running. Start it with 'make airflow-start'"; \
	fi

airflow-status:
	@echo "=== Docker Service Status ==="
	@systemctl is-active docker || echo "Docker service is not running. Start with: sudo systemctl start docker"
	@echo ""
	@echo "=== Container Status ==="
	@$(DOCKER_COMPOSE) ps || echo "No containers running"
	@echo ""
	@echo "=== Quick Fix Commands ==="
	@echo "1. Start Docker: sudo systemctl start docker"
	@echo "2. Start databases: make db-up"  
	@echo "3. Initialize Airflow: make airflow-init-manual"
	@echo "4. Start Airflow: make airflow-start"

# Pipeline Testing Commands
test-data-pipeline:
	@if docker ps | grep -q airflow-webserver; then \
		echo "Testing data collection pipeline..."; \
		docker exec ai-trading-advisor-airflow-webserver-1 airflow dags test data_collection_pipeline 2024-01-01; \
	else \
		echo "Airflow webserver container is not running. Start it with 'make airflow-start'"; \
	fi

test-analysis-pipeline:
	@if docker ps | grep -q airflow-webserver; then \
		echo "Testing analysis pipeline..."; \
		docker exec ai-trading-advisor-airflow-webserver-1 airflow dags test analysis_pipeline 2024-01-01; \
	else \
		echo "Airflow webserver container is not running. Start it with 'make airflow-start'"; \
	fi

test-recommendation-pipeline:
	@if docker ps | grep -q airflow-webserver; then \
		echo "Testing recommendation pipeline..."; \
		docker exec ai-trading-advisor-airflow-webserver-1 airflow dags test recommendation_pipeline 2024-01-01; \
	else \
		echo "Airflow webserver container is not running. Start it with 'make airflow-start'"; \
	fi

test-all-pipelines:
	@echo "Testing all Airflow pipelines..."
	make test-data-pipeline
	@echo ""
	make test-analysis-pipeline
	@echo ""
	make test-recommendation-pipeline

# DAG Trigger Commands
trigger-data-pipeline:
	@if docker ps | grep -q airflow-webserver; then \
		docker exec ai-trading-advisor-airflow-webserver-1 airflow dags trigger data_collection_pipeline; \
	else \
		echo "Airflow webserver container is not running. Start it with 'make airflow-start'"; \
	fi

trigger-simple-pipeline:
	@if docker ps | grep -q airflow-webserver; then \
		docker exec ai-trading-advisor-airflow-webserver-1 airflow dags trigger simple_data_collection_pipeline; \
	else \
		echo "Airflow webserver container is not running. Start it with 'make airflow-start'"; \
	fi

trigger-and-wait:
	@echo "Running final fix for AI Trading Advisor..."
	@./final_fix.sh

# Pipeline Monitoring Commands
airflow-task-list:
	@if docker ps | grep -q airflow-webserver; then \
		echo "=== Data Collection Pipeline Tasks ==="; \
		docker exec ai-trading-advisor-airflow-webserver-1 airflow tasks list data_collection_pipeline; \
		echo ""; \
		echo "=== Simple Data Collection Pipeline Tasks ==="; \
		docker exec ai-trading-advisor-airflow-webserver-1 airflow tasks list simple_data_collection_pipeline; \
	else \
		echo "Airflow webserver container is not running. Start it with 'make airflow-start'"; \
	fi

airflow-dag-state:
	@if docker ps | grep -q airflow-webserver; then \
		echo "=== DAG Run States ==="; \
		docker exec ai-trading-advisor-airflow-webserver-1 airflow dags state data_collection_pipeline 2024-01-01 || echo "No runs found"; \
		docker exec ai-trading-advisor-airflow-webserver-1 airflow dags state simple_data_collection_pipeline 2024-01-01 || echo "No runs found"; \
	else \
		echo "Airflow webserver container is not running. Start it with 'make airflow-start'"; \
	fi

# Quick pipeline validation
validate-dags:
	@if docker ps | grep -q airflow-webserver; then \
		echo "Validating all DAGs..."; \
		docker exec ai-trading-advisor-airflow-webserver-1 python -c "import sys; sys.path.append('/opt/airflow/src'); from airflow.models import DagBag; db = DagBag('/opt/airflow/dags'); print(f'DAGs loaded: {len(db.dags)}'); print(f'Import errors: {len(db.import_errors)}'); [print(f'Error in {f}: {e}') for f, e in db.import_errors.items()]"; \
	else \
		echo "Airflow webserver container is not running. Start it with 'make airflow-start'"; \
	fi

# Database Commands
db-up:
	$(DOCKER_COMPOSE) up -d postgres redis

db-reset:
	$(DOCKER_COMPOSE) down -v
	$(DOCKER_COMPOSE) up -d postgres redis

db-connect:
	@echo "Connecting to PostgreSQL database..."
	@docker exec -it ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor || echo "Use 'docker exec -it ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor' for interactive connection"

db-query:
	@echo "Running database query to show DAG outputs..."
	docker exec ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "\dt"

db-show-tables:
	@echo "=== Database Tables ==="
	@docker exec ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "\dt" || echo "Database not accessible"

db-show-data:
	@echo "=== Sample Data from Tables ==="
	@echo "Market Data:"
	@docker exec -i ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "SELECT COUNT(*) as count, symbol, price, volume FROM market_data GROUP BY symbol, price, volume ORDER BY symbol LIMIT 5;" 2>/dev/null || echo "No market_data table found, zero data"
	@echo ""
	@echo "News Data:"
	@docker exec -i ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "SELECT COUNT(*) as count, title, sentiment FROM news_data GROUP BY title, sentiment ORDER BY title LIMIT 5;" 2>/dev/null || echo "No news_data table found, zero data"
	@echo ""
	@echo "Analysis Results:"
	@docker exec -i ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "SELECT COUNT(*) as count, symbol, analysis_type FROM analysis_results GROUP BY symbol, analysis_type ORDER BY symbol LIMIT 5;" 2>/dev/null || echo "No analysis_results table found, zero data"
	@echo ""
	@echo "Recommendations:"
	@docker exec -i ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "SELECT COUNT(*) as count, symbol, action, confidence FROM recommendations GROUP BY symbol, action, confidence ORDER BY symbol LIMIT 5;" 2>/dev/null || echo "No recommendations table found, zero data"

db-create-tables:
	@echo "Creating database tables for DAG outputs..."
	docker exec ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "CREATE TABLE IF NOT EXISTS market_data (id SERIAL PRIMARY KEY, symbol VARCHAR(10), price DECIMAL, volume BIGINT, timestamp TIMESTAMP, execution_date DATE);"
	docker exec ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "CREATE TABLE IF NOT EXISTS news_data (id SERIAL PRIMARY KEY, title TEXT, content TEXT, sentiment DECIMAL, timestamp TIMESTAMP, execution_date DATE);"
	docker exec ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "CREATE TABLE IF NOT EXISTS analysis_results (id SERIAL PRIMARY KEY, symbol VARCHAR(10), analysis_type VARCHAR(50), results JSONB, timestamp TIMESTAMP, execution_date DATE);"
	docker exec ai-trading-advisor-postgres-1 psql -U trader -d trading_advisor -c "CREATE TABLE IF NOT EXISTS recommendations (id SERIAL PRIMARY KEY, symbol VARCHAR(10), action VARCHAR(20), confidence DECIMAL, position_size DECIMAL, risk_level VARCHAR(20), timestamp TIMESTAMP, execution_date DATE);"
	@echo "Database tables created successfully"

# Health Checks
health:
	@echo "Checking API health..."
	@curl -f http://localhost:8000/health || echo "API not responding"
	@echo "\nChecking Airflow health..."
	@curl -f http://localhost:8080/health || echo "Airflow not responding"

health-check:
	@echo "=== Comprehensive System Health Check ==="
	@echo ""
	@echo "=== Docker Service Status ==="
	@systemctl is-active docker || echo "❌ Docker service is not running"
	@echo ""
	@echo "=== Container Status ==="
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(ai-trading|NAMES)" || echo "No containers running"
	@echo ""
	@echo "=== API Health ==="
	@curl -s -f http://localhost:8000/health && echo "✅ API is healthy" || echo "❌ API not responding"
	@echo ""
	@echo "=== Airflow Health ==="  
	@curl -s -f http://localhost:8080/health && echo "✅ Airflow is healthy" || echo "❌ Airflow not responding"
	@echo ""
	@echo "=== Database Health ==="
	@docker exec ai-trading-advisor-postgres-1 pg_isready -U trader -d trading_advisor 2>/dev/null && echo "✅ PostgreSQL is healthy" || echo "❌ PostgreSQL not responding"
	@echo ""
	@echo "=== Redis Health ==="
	@docker exec ai-trading-advisor-redis-1 redis-cli ping 2>/dev/null | grep -q PONG && echo "✅ Redis is healthy" || echo "❌ Redis not responding"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

clean-all: clean
	rm -rf $(VENV)
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

# Help
help:
	@echo "Available commands:"
	@echo "  setup          - Initial project setup"
	@echo "  dev            - Start development server"
	@echo "  test           - Run tests"
	@echo "  test-all       - Run all tests"
	@echo "  test-real-data - Run real data integration tests (requires USE_REAL_DATA=True)"
	@echo "  test-coverage  - Run tests with coverage"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo "  docker-build   - Build Docker containers"
	@echo "  docker-up      - Start all Docker services"
	@echo "  docker-down    - Stop all Docker services"
	@echo "  airflow-init   - Initialize Airflow database"
	@echo "  airflow-init-manual - Manual Airflow initialization"
	@echo "  airflow-start  - Start Airflow services"
	@echo "  airflow-status - Check Airflow system status"
	@echo "  airflow-db-check - Check Airflow database"
	@echo "  airflow-dags-list - List Airflow DAGs"
	@echo "  test-data-pipeline - Test data collection pipeline"
	@echo "  test-analysis-pipeline - Test analysis pipeline"
	@echo "  test-recommendation-pipeline - Test recommendation pipeline"
	@echo "  test-all-pipelines - Test all pipelines"
	@echo "  trigger-data-pipeline - Trigger data collection pipeline"
	@echo "  trigger-simple-pipeline - Trigger simple data pipeline"
	@echo "  airflow-task-list - List tasks in pipelines"
	@echo "  airflow-dag-state - Check DAG run states"
	@echo "  validate-dags - Validate all DAGs for import errors"
	@echo "  db-up          - Start database services only"
	@echo "  db-connect     - Connect to PostgreSQL database interactively"
	@echo "  db-show-tables - Show all database tables"
	@echo "  db-show-data   - Show sample data from all tables"
	@echo "  db-create-tables - Create tables for DAG outputs"
	@echo "  health         - Check service health"
	@echo "  health-check   - Comprehensive system health check"
	@echo "  clean          - Clean temporary files"
