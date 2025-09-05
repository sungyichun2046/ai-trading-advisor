"""Technical and fundamental analysis pipeline DAG for AI Trading Advisor."""

from datetime import datetime, timedelta
from typing import Dict, List

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

from src.config import settings

# Default arguments for the DAG
default_args = {
    "owner": "trading-team",
    "depends_on_past": True,
    "start_date": days_ago(1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=30),
}

# Define the DAG
dag = DAG(
    "analysis_pipeline",
    default_args=default_args,
    description="Perform technical and fundamental analysis on collected market data",
    schedule_interval="*/30 * * * *",  # Every 30 minutes
    catchup=False,
    max_active_runs=1,
    tags=["analysis", "technical", "fundamental"],
)


def perform_technical_analysis(**context) -> Dict:
    """Perform technical analysis on market data.

    Returns:
        Dict: Technical analysis results and indicators
    """
    from src.core.analysis_engine import TechnicalAnalysisEngine

    execution_date = context["execution_date"]
    engine = TechnicalAnalysisEngine()

    # Define symbols to analyze
    symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    results = {}
    for symbol in symbols:
        try:
            # Calculate technical indicators
            indicators = engine.calculate_indicators(symbol)

            # Generate technical signals
            signals = engine.generate_signals(symbol, indicators)

            results[symbol] = {
                "status": "success",
                "indicators": indicators,
                "signals": signals,
                "trend": engine.determine_trend(indicators),
                "strength": engine.calculate_trend_strength(indicators),
                "timestamp": execution_date.isoformat(),
            }
        except Exception as e:
            results[symbol] = {
                "status": "failed",
                "error": str(e),
                "timestamp": execution_date.isoformat(),
            }

    return results


def perform_fundamental_analysis(**context) -> Dict:
    """Perform fundamental analysis on securities.

    Returns:
        Dict: Fundamental analysis results
    """
    from src.core.analysis_engine import FundamentalAnalysisEngine

    execution_date = context["execution_date"]
    engine = FundamentalAnalysisEngine()

    # Focus on individual stocks for fundamental analysis
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

    results = {}
    for symbol in symbols:
        try:
            # Get fundamental data
            fundamentals = engine.get_fundamental_data(symbol)

            # Calculate financial ratios
            ratios = engine.calculate_financial_ratios(fundamentals)

            # Perform valuation analysis
            valuation = engine.perform_valuation_analysis(symbol, fundamentals)

            results[symbol] = {
                "status": "success",
                "fundamentals": fundamentals,
                "ratios": ratios,
                "valuation": valuation,
                "score": engine.calculate_fundamental_score(ratios, valuation),
                "timestamp": execution_date.isoformat(),
            }
        except Exception as e:
            results[symbol] = {
                "status": "failed",
                "error": str(e),
                "timestamp": execution_date.isoformat(),
            }

    return results


def perform_sentiment_analysis(**context) -> Dict:
    """Analyze market sentiment from news and social media.

    Returns:
        Dict: Sentiment analysis results
    """
    from src.core.analysis_engine import SentimentAnalysisEngine

    execution_date = context["execution_date"]
    engine = SentimentAnalysisEngine()

    try:
        # Analyze overall market sentiment
        market_sentiment = engine.analyze_market_sentiment()

        # Analyze sector-specific sentiment
        sector_sentiment = engine.analyze_sector_sentiment()

        # Analyze individual stock sentiment
        stock_sentiment = engine.analyze_stock_sentiment(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        )

        # Calculate sentiment momentum
        sentiment_momentum = engine.calculate_sentiment_momentum()

        return {
            "status": "success",
            "market_sentiment": market_sentiment,
            "sector_sentiment": sector_sentiment,
            "stock_sentiment": stock_sentiment,
            "momentum": sentiment_momentum,
            "timestamp": execution_date.isoformat(),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


def perform_risk_analysis(**context) -> Dict:
    """Perform comprehensive risk analysis.

    Returns:
        Dict: Risk analysis results
    """
    from src.core.risk_engine import RiskAnalysisEngine

    execution_date = context["execution_date"]
    engine = RiskAnalysisEngine()

    try:
        # Calculate market risk metrics
        market_risk = engine.calculate_market_risk()

        # Calculate volatility metrics
        volatility_metrics = engine.calculate_volatility_metrics()

        # Perform correlation analysis
        correlation_analysis = engine.perform_correlation_analysis()

        # Calculate Value at Risk (VaR)
        var_analysis = engine.calculate_var_metrics()

        # Assess tail risk
        tail_risk = engine.assess_tail_risk()

        return {
            "status": "success",
            "market_risk": market_risk,
            "volatility": volatility_metrics,
            "correlations": correlation_analysis,
            "var_metrics": var_analysis,
            "tail_risk": tail_risk,
            "risk_score": engine.calculate_overall_risk_score(),
            "timestamp": execution_date.isoformat(),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


def generate_analysis_summary(**context) -> Dict:
    """Generate comprehensive analysis summary.

    Returns:
        Dict: Combined analysis summary
    """
    from src.core.analysis_engine import AnalysisSummaryEngine

    execution_date = context["execution_date"]
    engine = AnalysisSummaryEngine()

    # Get results from previous tasks
    technical_results = context["task_instance"].xcom_pull(
        task_ids="technical_analysis.perform_technical_analysis"
    )
    fundamental_results = context["task_instance"].xcom_pull(
        task_ids="fundamental_analysis.perform_fundamental_analysis"
    )
    sentiment_results = context["task_instance"].xcom_pull(
        task_ids="sentiment_analysis.perform_sentiment_analysis"
    )
    risk_results = context["task_instance"].xcom_pull(
        task_ids="risk_analysis.perform_risk_analysis"
    )

    try:
        # Combine all analysis results
        summary = engine.create_comprehensive_summary(
            technical_results, fundamental_results, sentiment_results, risk_results
        )

        # Generate market outlook
        market_outlook = engine.generate_market_outlook(summary)

        # Identify key opportunities and risks
        opportunities = engine.identify_opportunities(summary)
        risks = engine.identify_risks(summary)

        return {
            "status": "success",
            "summary": summary,
            "market_outlook": market_outlook,
            "opportunities": opportunities,
            "risks": risks,
            "confidence_score": engine.calculate_confidence_score(summary),
            "timestamp": execution_date.isoformat(),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


def store_analysis_results(**context) -> Dict:
    """Store analysis results in database.

    Returns:
        Dict: Storage operation results
    """
    from src.data.database import AnalysisDataManager

    db_manager = AnalysisDataManager()
    execution_date = context["execution_date"]

    # Get all analysis results
    technical_results = context["task_instance"].xcom_pull(
        task_ids="technical_analysis.perform_technical_analysis"
    )
    fundamental_results = context["task_instance"].xcom_pull(
        task_ids="fundamental_analysis.perform_fundamental_analysis"
    )
    sentiment_results = context["task_instance"].xcom_pull(
        task_ids="sentiment_analysis.perform_sentiment_analysis"
    )
    risk_results = context["task_instance"].xcom_pull(
        task_ids="risk_analysis.perform_risk_analysis"
    )
    summary_results = context["task_instance"].xcom_pull(
        task_ids="generate_analysis_summary"
    )

    try:
        # Store technical analysis
        technical_storage = db_manager.store_technical_analysis(
            technical_results, execution_date
        )

        # Store fundamental analysis
        fundamental_storage = db_manager.store_fundamental_analysis(
            fundamental_results, execution_date
        )

        # Store sentiment analysis
        sentiment_storage = db_manager.store_sentiment_analysis(
            sentiment_results, execution_date
        )

        # Store risk analysis
        risk_storage = db_manager.store_risk_analysis(risk_results, execution_date)

        # Store analysis summary
        summary_storage = db_manager.store_analysis_summary(
            summary_results, execution_date
        )

        return {
            "status": "success",
            "technical_records": technical_storage.get("count", 0),
            "fundamental_records": fundamental_storage.get("count", 0),
            "sentiment_records": sentiment_storage.get("count", 0),
            "risk_records": risk_storage.get("count", 0),
            "summary_records": summary_storage.get("count", 0),
            "timestamp": execution_date.isoformat(),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


# Task definitions
with dag:
    # Wait for data collection to complete
    wait_for_data = ExternalTaskSensor(
        task_id="wait_for_data_collection",
        external_dag_id="data_collection_pipeline",
        external_task_id="pipeline_health_check",
        timeout=300,
        poke_interval=30,
        doc_md="""
        ### Wait for Data Collection
        
        Waits for the data collection pipeline to complete before starting analysis.
        
        **Dependency**: data_collection_pipeline.pipeline_health_check
        **Timeout**: 5 minutes
        """,
    )

    # Technical analysis task group
    with TaskGroup(
        "technical_analysis", tooltip="Technical analysis tasks"
    ) as technical_analysis:
        technical_task = PythonOperator(
            task_id="perform_technical_analysis",
            python_callable=perform_technical_analysis,
            doc_md="""
            ### Technical Analysis
            
            Performs comprehensive technical analysis:
            - Moving averages (SMA, EMA)
            - Momentum indicators (RSI, MACD, Stochastic)
            - Trend indicators (ADX, Bollinger Bands)
            - Volume indicators (OBV, VWAP)
            - Support/resistance levels
            """,
        )

    # Fundamental analysis task group
    with TaskGroup(
        "fundamental_analysis", tooltip="Fundamental analysis tasks"
    ) as fundamental_analysis:
        fundamental_task = PythonOperator(
            task_id="perform_fundamental_analysis",
            python_callable=perform_fundamental_analysis,
            doc_md="""
            ### Fundamental Analysis
            
            Performs fundamental analysis:
            - Financial ratios (P/E, P/B, ROE, ROA)
            - Growth metrics
            - Profitability analysis
            - Valuation models (DCF, comparative)
            - Financial health assessment
            """,
        )

    # Sentiment analysis task group
    with TaskGroup(
        "sentiment_analysis", tooltip="Sentiment analysis tasks"
    ) as sentiment_analysis:
        sentiment_task = PythonOperator(
            task_id="perform_sentiment_analysis",
            python_callable=perform_sentiment_analysis,
            doc_md="""
            ### Sentiment Analysis
            
            Analyzes market sentiment:
            - News sentiment scoring
            - Social media sentiment
            - Market momentum indicators
            - Fear & greed index
            - Sector rotation analysis
            """,
        )

    # Risk analysis task group
    with TaskGroup("risk_analysis", tooltip="Risk analysis tasks") as risk_analysis:
        risk_task = PythonOperator(
            task_id="perform_risk_analysis",
            python_callable=perform_risk_analysis,
            doc_md="""
            ### Risk Analysis
            
            Comprehensive risk assessment:
            - Value at Risk (VaR) calculations
            - Volatility analysis
            - Correlation analysis
            - Tail risk assessment
            - Market risk metrics
            """,
        )

    # Analysis summary task
    summary_task = PythonOperator(
        task_id="generate_analysis_summary",
        python_callable=generate_analysis_summary,
        doc_md="""
        ### Analysis Summary
        
        Combines all analysis types into comprehensive summary:
        - Market outlook generation
        - Opportunity identification
        - Risk assessment summary
        - Confidence scoring
        """,
    )

    # Storage task
    storage_task = PythonOperator(
        task_id="store_analysis_results",
        python_callable=store_analysis_results,
        doc_md="""
        ### Store Analysis Results
        
        Stores all analysis results in database:
        - Technical indicators and signals
        - Fundamental scores and ratios
        - Sentiment metrics
        - Risk assessments
        - Analysis summaries
        """,
    )

    # Final health check
    analysis_health_check = BashOperator(
        task_id="analysis_health_check",
        bash_command="""
        echo "Analysis pipeline completed successfully"
        echo "Execution date: {{ ds }}"
        echo "Summary: {{ task_instance.xcom_pull(task_ids='generate_analysis_summary') }}"
        """,
        doc_md="""
        ### Analysis Health Check
        
        Final verification of analysis pipeline:
        - Confirms successful completion
        - Logs analysis metrics
        - Prepares for recommendation pipeline
        """,
    )

    # Define task dependencies
    (
        wait_for_data
        >> [technical_analysis, fundamental_analysis, sentiment_analysis, risk_analysis]
        >> summary_task
        >> storage_task
        >> analysis_health_check
    )
