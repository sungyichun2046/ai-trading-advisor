"""Simplified data collection DAG for AI Trading Advisor."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

def collect_market_data(**context):
    """Mock market data collection."""
    print(f"Collecting market data at {datetime.now()}")
    return "Market data collected successfully"

def collect_news_data(**context):
    """Mock news data collection."""
    print(f"Collecting news data at {datetime.now()}")
    return "News data collected successfully"

def validate_data(**context):
    """Mock data validation."""
    print(f"Validating collected data at {datetime.now()}")
    return "Data validation completed"

# Default arguments
default_args = {
    "owner": "trading-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "data_collection_dag",
    default_args=default_args,
    description="Data collection pipeline for market data and news",
    schedule_interval=timedelta(minutes=15),
    catchup=False,
    tags=["data", "collection", "market"],
)

# Define tasks
market_data_task = PythonOperator(
    task_id="collect_market_data",
    python_callable=collect_market_data,
    dag=dag,
)

news_data_task = PythonOperator(
    task_id="collect_news_data", 
    python_callable=collect_news_data,
    dag=dag,
)

validate_data_task = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag,
)

# Set task dependencies
[market_data_task, news_data_task] >> validate_data_task