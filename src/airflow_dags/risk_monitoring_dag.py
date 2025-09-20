"""Simplified risk monitoring DAG for AI Trading Advisor."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

def monitor_portfolio_risk(**context):
    """Mock portfolio risk monitoring."""
    print(f"Monitoring portfolio risk at {datetime.now()}")
    return "Portfolio risk monitored"

def check_position_limits(**context):
    """Mock position limit checking."""
    print(f"Checking position limits at {datetime.now()}")
    return "Position limits checked"

def calculate_var(**context):
    """Mock VaR calculation."""
    print(f"Calculating Value at Risk at {datetime.now()}")
    return "VaR calculated"

def generate_risk_report(**context):
    """Mock risk report generation."""
    print(f"Generating risk report at {datetime.now()}")
    return "Risk report generated"

# Default arguments
default_args = {
    "owner": "risk-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "risk_monitoring_dag",
    default_args=default_args,
    description="Risk monitoring and compliance pipeline",
    schedule_interval=timedelta(minutes=30),
    catchup=False,
    tags=["risk", "monitoring", "compliance"],
)

# Define tasks
portfolio_risk_task = PythonOperator(
    task_id="monitor_portfolio_risk",
    python_callable=monitor_portfolio_risk,
    dag=dag,
)

position_limits_task = PythonOperator(
    task_id="check_position_limits",
    python_callable=check_position_limits,
    dag=dag,
)

var_task = PythonOperator(
    task_id="calculate_var",
    python_callable=calculate_var,
    dag=dag,
)

report_task = PythonOperator(
    task_id="generate_risk_report",
    python_callable=generate_risk_report,
    dag=dag,
)

# Set task dependencies
[portfolio_risk_task, position_limits_task, var_task] >> report_task