"""Simplified analysis DAG for AI Trading Advisor."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

def technical_analysis(**context):
    """Mock technical analysis."""
    print(f"Running technical analysis at {datetime.now()}")
    return "Technical analysis completed"

def fundamental_analysis(**context):
    """Mock fundamental analysis."""
    print(f"Running fundamental analysis at {datetime.now()}")
    return "Fundamental analysis completed"

def sentiment_analysis(**context):
    """Mock sentiment analysis."""
    print(f"Running sentiment analysis at {datetime.now()}")
    return "Sentiment analysis completed"

def generate_signals(**context):
    """Mock signal generation."""
    print(f"Generating trading signals at {datetime.now()}")
    return "Trading signals generated"

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
    "analysis_dag",
    default_args=default_args,
    description="Analysis pipeline for market data",
    schedule_interval=timedelta(minutes=30),
    catchup=False,
    tags=["analysis", "signals", "trading"],
)

# Define tasks
technical_task = PythonOperator(
    task_id="technical_analysis",
    python_callable=technical_analysis,
    dag=dag,
)

fundamental_task = PythonOperator(
    task_id="fundamental_analysis",
    python_callable=fundamental_analysis,
    dag=dag,
)

sentiment_task = PythonOperator(
    task_id="sentiment_analysis",
    python_callable=sentiment_analysis,
    dag=dag,
)

signals_task = PythonOperator(
    task_id="generate_signals",
    python_callable=generate_signals,
    dag=dag,
)

# Set task dependencies
[technical_task, fundamental_task, sentiment_task] >> signals_task