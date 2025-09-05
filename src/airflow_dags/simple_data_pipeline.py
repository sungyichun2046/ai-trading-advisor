"""Simplified data collection pipeline DAG for AI Trading Advisor."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Default arguments for the DAG
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
    "simple_data_collection_pipeline",
    default_args=default_args,
    description="Simplified data collection pipeline without external dependencies",
    schedule_interval="*/15 * * * *",  # Every 15 minutes
    catchup=False,
    max_active_runs=1,
    tags=["data", "collection", "simple"],
)


def collect_sample_data(**context):
    """Collect sample market data (placeholder)."""
    import logging

    logger = logging.getLogger(__name__)

    execution_date = context["execution_date"]
    symbols = ["SPY", "QQQ", "AAPL", "MSFT"]

    results = {}
    for symbol in symbols:
        results[symbol] = {
            "status": "success",
            "price": 100.0,
            "volume": 1000000,
            "timestamp": execution_date.isoformat(),
        }

    logger.info(f"Collected data for {len(symbols)} symbols")
    return results


def validate_sample_data(**context):
    """Validate collected data (placeholder)."""
    import logging

    logger = logging.getLogger(__name__)

    # Get data from previous task
    data = context["task_instance"].xcom_pull(task_ids="collect_data")

    if not data:
        raise ValueError("No data to validate")

    logger.info(f"Validated {len(data)} symbols")
    return True


def store_sample_data(**context):
    """Store validated data (placeholder)."""
    import logging

    logger = logging.getLogger(__name__)

    data = context["task_instance"].xcom_pull(task_ids="collect_data")
    execution_date = context["execution_date"]

    logger.info(f"Stored {len(data) if data else 0} symbols at {execution_date}")
    return {
        "status": "success",
        "records_stored": len(data) if data else 0,
        "timestamp": execution_date.isoformat(),
    }


# Task definitions
with dag:
    collect_task = PythonOperator(
        task_id="collect_data",
        python_callable=collect_sample_data,
        doc_md="""
        ### Collect Sample Data
        
        Collects sample market data for testing purposes.
        Returns mock data for major ETFs and stocks.
        """,
    )

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_sample_data,
        doc_md="""
        ### Validate Data
        
        Validates that data was collected successfully.
        Checks for data completeness and format.
        """,
    )

    store_task = PythonOperator(
        task_id="store_data",
        python_callable=store_sample_data,
        doc_md="""
        ### Store Data
        
        Stores validated data (placeholder implementation).
        In production, this would write to PostgreSQL.
        """,
    )

    health_check = BashOperator(
        task_id="health_check",
        bash_command='echo "Simple data pipeline completed at {{ ds }}"',
        doc_md="""
        ### Health Check
        
        Simple health check to confirm pipeline completion.
        """,
    )

    # Define task dependencies
    collect_task >> validate_task >> store_task >> health_check
