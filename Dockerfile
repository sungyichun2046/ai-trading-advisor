# Multi-stage build for better dependency management
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# FastAPI Application Stage
FROM base as fastapi

WORKDIR /app

# Install Python dependencies (including dev dependencies for testing)
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash trader && \
    chown -R trader:trader /app
USER trader

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Airflow Stage  
FROM base as airflow

WORKDIR /opt/airflow

# Install Airflow dependencies
COPY requirements-airflow.txt .
RUN pip install --no-cache-dir -r requirements-airflow.txt

# Copy application code for DAGs
COPY . /app/
COPY src/airflow_dags /opt/airflow/dags/

# Set Airflow environment
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

# Create airflow user
RUN useradd --create-home --shell /bin/bash airflow && \
    chown -R airflow:airflow /opt/airflow
USER airflow

# Default Airflow command
CMD ["airflow", "webserver"]