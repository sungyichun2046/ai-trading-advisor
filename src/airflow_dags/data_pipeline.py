"""Data collection pipeline DAG for AI Trading Advisor."""

from datetime import datetime, timedelta
from typing import List

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
# from airflow.sensors.filesystem import FileSensor  # Not used in current implementation
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

from src.config import settings
import logging

logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    "owner": "trading-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

# Define the DAG
dag = DAG(
    "data_collection_pipeline",
    default_args=default_args,
    description="Collect and process market data from multiple sources",
    schedule_interval="*/15 * * * *",  # Every 15 minutes during market hours
    catchup=False,
    max_active_runs=1,
    tags=["data", "collection", "market"],
)


def collect_market_data(**context) -> dict:
    """Collect real-time market data from multiple sources.

    Returns:
        dict: Collection status and metadata
    """
    from src.data.collectors import MarketDataCollector

    execution_date = context["execution_date"]
    collector = MarketDataCollector()

    # Define symbols to collect (TODO: move to config.py)
    symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    results = {}
    for symbol in symbols:
        try:
            data = collector.collect_real_time_data(symbol)
            if data:
                results[symbol] = data
            else:
                results[symbol] = {
                    "status": "failed",
                    "error": "No data returned",
                    "symbol": symbol,
                    "timestamp": execution_date.isoformat(),
                }
        except Exception as e:
            results[symbol] = {
                "status": "failed",
                "error": str(e),
                "symbol": symbol,
                "timestamp": execution_date.isoformat(),
            }

    logger.info(f"Collected data for {len(results)} symbols")
    return results


def collect_news_sentiment(**context) -> dict:
    """Collect and analyze news sentiment data.

    Returns:
        dict: Sentiment analysis results
    """
    from src.data.collectors import NewsCollector

    execution_date = context["execution_date"]
    collector = NewsCollector()

    try:
        # Collect financial news
        news_articles = collector.collect_financial_news()

        # Analyze sentiment
        sentiment_results = collector.analyze_sentiment(news_articles)

        return {
            "status": "success",
            "timestamp": execution_date.isoformat(),
            "articles_count": len(news_articles),
            **sentiment_results  # Include all sentiment analysis results
        }
    except Exception as e:
        logger.error(f"Failed to collect news sentiment: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


def validate_data_quality(**context) -> bool:
    """Validate collected data quality and completeness.

    Returns:
        bool: True if data quality is acceptable
    """
    from src.data.processors import DataValidator

    validator = DataValidator()

    # Get data from previous tasks
    market_data_results = context["task_instance"].xcom_pull(
        task_ids="collect_market_data"
    )
    news_results = context["task_instance"].xcom_pull(task_ids="collect_news_sentiment")

    # Validate market data
    market_validation = validator.validate_market_data(market_data_results)

    # Validate news data
    news_validation = validator.validate_news_data(news_results)

    # Check if validation passes minimum thresholds
    is_valid = (
        market_validation.get("completeness", 0) >= 0.8
        and market_validation.get("freshness", 0) >= 0.9
        and news_validation.get("coverage", 0) >= 0.7
    )

    if not is_valid:
        raise ValueError(
            f"Data quality validation failed: {market_validation}, {news_validation}"
        )

    return True


def initialize_database(**context) -> dict:
    """Initialize database tables if they don't exist.

    Returns:
        dict: Initialization results
    """
    from src.data.database import DatabaseManager

    try:
        db_manager = DatabaseManager()
        db_manager.create_tables()
        logger.info("Database tables initialized successfully")
        return {
            "status": "success",
            "message": "Database tables created/verified",
            "timestamp": context["execution_date"].isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": context["execution_date"].isoformat()
        }


def store_processed_data(**context) -> dict:
    """Store validated data in database.

    Returns:
        dict: Storage operation results
    """
    from src.data.database import MarketDataStorage, NewsStorage

    execution_date = context["execution_date"]

    # Get validated data from previous tasks
    market_data = context["task_instance"].xcom_pull(task_ids="collect_market_data")
    news_data = context["task_instance"].xcom_pull(task_ids="collect_news_sentiment")

    try:
        # Store market data
        market_storage = MarketDataStorage()
        market_storage_result = market_storage.store_market_data(
            market_data, execution_date
        )

        # Store news sentiment data
        news_storage = NewsStorage()
        news_storage_result = news_storage.store_news_data(news_data, execution_date)

        return {
            "status": "success",
            "market_records_stored": market_storage_result.get("count", 0),
            "news_records_stored": news_storage_result.get("count", 0),
            "timestamp": execution_date.isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to store data: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


# Task definitions
with dag:
    # Database initialization
    init_db_task = PythonOperator(
        task_id="initialize_database",
        python_callable=initialize_database,
        doc_md="""
        ### Initialize Database
        
        Creates database tables if they don't exist:
        - market_data table with indexes
        - news_data table with sentiment scores
        - analysis_results table for future use
        - recommendations table for future use
        """,
    )

    # Data collection task group
    with TaskGroup(
        "data_collection", tooltip="Collect data from various sources"
    ) as data_collection:
        collect_market_task = PythonOperator(
            task_id="collect_market_data",
            python_callable=collect_market_data,
            doc_md="""
            ### Collect Market Data
            
            Collects real-time market data using yfinance:
            - Price data (OHLCV)
            - Volume data
            - Market cap and P/E ratios
            - 15-minute intervals
            
            **Sources**: yfinance (real-time) or dummy data (development)
            **Frequency**: Every 15 minutes during market hours
            **Symbols**: Major ETFs and stocks
            """,
        )

        collect_news_task = PythonOperator(
            task_id="collect_news_sentiment",
            python_callable=collect_news_sentiment,
            doc_md="""
            ### Collect News & Sentiment
            
            Collects financial news and performs sentiment analysis:
            - Financial news articles from NewsAPI
            - FinBERT sentiment analysis
            - Market impact assessment
            
            **Sources**: NewsAPI (real) or dummy data (development)
            **Analysis**: FinBERT model for financial sentiment
            """,
        )

    # Data validation task
    validate_task = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validate_data_quality,
        doc_md="""
        ### Data Quality Validation
        
        Validates collected data for:
        - Completeness (>80% of expected data points)
        - Freshness (>90% within acceptable time window)
        - News coverage (>70% of market hours covered)
        
        **Failure handling**: Raises exception if quality thresholds not met
        """,
    )

    # Data storage task
    store_task = PythonOperator(
        task_id="store_processed_data",
        python_callable=store_processed_data,
        doc_md="""
        ### Store Processed Data
        
        Stores validated data in PostgreSQL database:
        - Market data with OHLCV + metadata
        - News sentiment with individual article scores
        - Proper indexing for query performance
        
        **Database**: PostgreSQL with optimized schema
        **Retention**: Configurable data retention policies
        """,
    )

    # Health check task
    health_check = BashOperator(
        task_id="pipeline_health_check",
        bash_command="""
        echo "Data pipeline completed successfully"
        echo "Execution date: {{ ds }}"
        echo "Market data status: {{ task_instance.xcom_pull(task_ids='collect_market_data')['SPY']['status'] if task_instance.xcom_pull(task_ids='collect_market_data') else 'No data' }}"
        echo "Storage status: {{ task_instance.xcom_pull(task_ids='store_processed_data')['status'] if task_instance.xcom_pull(task_ids='store_processed_data') else 'No data' }}"
        """,
        doc_md="""
        ### Pipeline Health Check
        
        Final health check and logging:
        - Confirms successful pipeline execution
        - Logs execution metrics
        - Reports data collection status
        """,
    )

    # Define task dependencies
    init_db_task >> data_collection >> validate_task >> store_task >> health_check
