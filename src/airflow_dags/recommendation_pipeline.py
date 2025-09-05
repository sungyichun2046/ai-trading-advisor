"""Trading recommendation pipeline DAG for AI Trading Advisor."""

from datetime import datetime, timedelta
from typing import Dict, List

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
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
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=20),
}

# Define the DAG
dag = DAG(
    "recommendation_pipeline",
    default_args=default_args,
    description="Generate trading recommendations based on comprehensive analysis",
    schedule_interval="0 */1 * * *",  # Every hour
    catchup=False,
    max_active_runs=1,
    tags=["recommendations", "trading", "signals"],
)


def generate_trading_signals(**context) -> Dict:
    """Generate trading signals based on analysis results.

    Returns:
        Dict: Trading signals and confidence scores
    """
    from src.core.recommendation_engine import TradingSignalEngine

    execution_date = context["execution_date"]
    engine = TradingSignalEngine()

    # Define symbols for signal generation
    symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    results = {}
    for symbol in symbols:
        try:
            # Generate buy/sell/hold signals
            signal = engine.generate_signal(symbol)

            # Calculate signal confidence
            confidence = engine.calculate_signal_confidence(symbol, signal)

            # Determine optimal entry/exit points
            entry_exit = engine.calculate_entry_exit_points(symbol, signal)

            results[symbol] = {
                "status": "success",
                "signal": signal,  # 'BUY', 'SELL', 'HOLD'
                "confidence": confidence,  # 0.0 to 1.0
                "entry_price": entry_exit.get("entry_price"),
                "exit_price": entry_exit.get("exit_price"),
                "stop_loss": entry_exit.get("stop_loss"),
                "target_price": entry_exit.get("target_price"),
                "risk_reward_ratio": entry_exit.get("risk_reward_ratio"),
                "timestamp": execution_date.isoformat(),
            }
        except Exception as e:
            results[symbol] = {
                "status": "failed",
                "error": str(e),
                "timestamp": execution_date.isoformat(),
            }

    return results


def calculate_position_sizes(**context) -> Dict:
    """Calculate optimal position sizes based on risk management rules.

    Returns:
        Dict: Position sizing recommendations
    """
    from src.core.risk_engine import PositionSizingEngine

    execution_date = context["execution_date"]
    engine = PositionSizingEngine()

    # Get trading signals from previous task
    trading_signals = context["task_instance"].xcom_pull(
        task_ids="signal_generation.generate_trading_signals"
    )

    try:
        # Get current portfolio status
        portfolio_status = engine.get_portfolio_status()

        results = {}
        for symbol, signal_data in trading_signals.items():
            if signal_data.get("status") == "success" and signal_data.get("signal") in [
                "BUY",
                "SELL",
            ]:
                try:
                    # Calculate position size based on risk management
                    position_size = engine.calculate_position_size(
                        symbol=symbol,
                        signal=signal_data["signal"],
                        confidence=signal_data["confidence"],
                        stop_loss=signal_data.get("stop_loss"),
                        portfolio_balance=portfolio_status["total_balance"],
                        current_positions=portfolio_status["positions"],
                    )

                    # Validate against risk limits
                    validation = engine.validate_position_size(
                        symbol, position_size, portfolio_status
                    )

                    results[symbol] = {
                        "status": "success",
                        "position_size": position_size,
                        "position_percentage": position_size
                        / portfolio_status["total_balance"],
                        "validation": validation,
                        "max_risk": validation.get("max_risk_amount"),
                        "timestamp": execution_date.isoformat(),
                    }
                except Exception as e:
                    results[symbol] = {
                        "status": "failed",
                        "error": str(e),
                        "timestamp": execution_date.isoformat(),
                    }

        return results
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


def generate_portfolio_recommendations(**context) -> Dict:
    """Generate portfolio-level recommendations.

    Returns:
        Dict: Portfolio optimization recommendations
    """
    from src.core.recommendation_engine import PortfolioOptimizationEngine

    execution_date = context["execution_date"]
    engine = PortfolioOptimizationEngine()

    # Get signals and position sizes from previous tasks
    trading_signals = context["task_instance"].xcom_pull(
        task_ids="signal_generation.generate_trading_signals"
    )
    position_sizes = context["task_instance"].xcom_pull(
        task_ids="position_sizing.calculate_position_sizes"
    )

    try:
        # Generate portfolio allocation recommendations
        allocation_recs = engine.generate_allocation_recommendations(
            trading_signals, position_sizes
        )

        # Calculate portfolio risk metrics
        portfolio_risk = engine.calculate_portfolio_risk(allocation_recs)

        # Generate rebalancing recommendations
        rebalancing_recs = engine.generate_rebalancing_recommendations()

        # Calculate expected portfolio returns
        expected_returns = engine.calculate_expected_returns(allocation_recs)

        # Generate diversification analysis
        diversification = engine.analyze_diversification(allocation_recs)

        return {
            "status": "success",
            "allocation_recommendations": allocation_recs,
            "portfolio_risk": portfolio_risk,
            "rebalancing_recommendations": rebalancing_recs,
            "expected_returns": expected_returns,
            "diversification_analysis": diversification,
            "portfolio_score": engine.calculate_portfolio_score(allocation_recs),
            "timestamp": execution_date.isoformat(),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


def apply_risk_filters(**context) -> Dict:
    """Apply risk management filters to recommendations.

    Returns:
        Dict: Filtered and validated recommendations
    """
    from src.core.risk_engine import RiskFilterEngine

    execution_date = context["execution_date"]
    engine = RiskFilterEngine()

    # Get all recommendation data
    trading_signals = context["task_instance"].xcom_pull(
        task_ids="signal_generation.generate_trading_signals"
    )
    position_sizes = context["task_instance"].xcom_pull(
        task_ids="position_sizing.calculate_position_sizes"
    )
    portfolio_recs = context["task_instance"].xcom_pull(
        task_ids="portfolio_optimization.generate_portfolio_recommendations"
    )

    try:
        # Apply risk filters
        filtered_signals = engine.filter_signals_by_risk(trading_signals)

        # Validate position sizes against risk limits
        validated_positions = engine.validate_positions_risk(position_sizes)

        # Check portfolio-level risk constraints
        portfolio_validation = engine.validate_portfolio_risk(portfolio_recs)

        # Apply market condition filters
        market_filters = engine.apply_market_condition_filters(filtered_signals)

        # Generate final recommendation list
        final_recommendations = engine.generate_final_recommendations(
            filtered_signals, validated_positions, portfolio_validation, market_filters
        )

        return {
            "status": "success",
            "filtered_signals": filtered_signals,
            "validated_positions": validated_positions,
            "portfolio_validation": portfolio_validation,
            "market_filters": market_filters,
            "final_recommendations": final_recommendations,
            "total_recommendations": len(final_recommendations),
            "risk_score": engine.calculate_overall_risk_score(final_recommendations),
            "timestamp": execution_date.isoformat(),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


def generate_recommendation_report(**context) -> Dict:
    """Generate comprehensive recommendation report.

    Returns:
        Dict: Formatted recommendation report
    """
    from src.core.recommendation_engine import ReportEngine

    execution_date = context["execution_date"]
    engine = ReportEngine()

    # Get all data from previous tasks
    trading_signals = context["task_instance"].xcom_pull(
        task_ids="signal_generation.generate_trading_signals"
    )
    position_sizes = context["task_instance"].xcom_pull(
        task_ids="position_sizing.calculate_position_sizes"
    )
    portfolio_recs = context["task_instance"].xcom_pull(
        task_ids="portfolio_optimization.generate_portfolio_recommendations"
    )
    risk_filtered = context["task_instance"].xcom_pull(
        task_ids="risk_management.apply_risk_filters"
    )

    try:
        # Generate executive summary
        executive_summary = engine.generate_executive_summary(risk_filtered)

        # Create detailed signal analysis
        signal_analysis = engine.create_signal_analysis(trading_signals)

        # Generate risk assessment summary
        risk_summary = engine.create_risk_summary(portfolio_recs, risk_filtered)

        # Create actionable recommendations
        actionable_recs = engine.create_actionable_recommendations(risk_filtered)

        # Generate performance projections
        performance_projections = engine.generate_performance_projections(
            portfolio_recs
        )

        # Create market context
        market_context = engine.create_market_context()

        return {
            "status": "success",
            "executive_summary": executive_summary,
            "signal_analysis": signal_analysis,
            "risk_summary": risk_summary,
            "actionable_recommendations": actionable_recs,
            "performance_projections": performance_projections,
            "market_context": market_context,
            "report_timestamp": execution_date.isoformat(),
            "confidence_level": engine.calculate_report_confidence(risk_filtered),
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": execution_date.isoformat(),
        }


def store_recommendations(**context) -> Dict:
    """Store recommendations in database and trigger notifications.

    Returns:
        Dict: Storage and notification results
    """
    from src.core.notification_engine import NotificationEngine
    from src.data.database import RecommendationDataManager

    db_manager = RecommendationDataManager()
    notification_engine = NotificationEngine()
    execution_date = context["execution_date"]

    # Get final report
    recommendation_report = context["task_instance"].xcom_pull(
        task_ids="generate_recommendation_report"
    )

    try:
        # Store recommendations in database
        storage_result = db_manager.store_recommendations(
            recommendation_report, execution_date
        )

        # Send notifications for high-priority recommendations
        notification_result = notification_engine.send_recommendation_notifications(
            recommendation_report
        )

        # Update recommendation tracking
        tracking_result = db_manager.update_recommendation_tracking(
            recommendation_report, execution_date
        )

        return {
            "status": "success",
            "stored_recommendations": storage_result.get("count", 0),
            "notifications_sent": notification_result.get("count", 0),
            "tracking_updated": tracking_result.get("success", False),
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
    # Wait for analysis to complete
    wait_for_analysis = ExternalTaskSensor(
        task_id="wait_for_analysis",
        external_dag_id="analysis_pipeline",
        external_task_id="analysis_health_check",
        timeout=600,
        poke_interval=60,
        doc_md="""
        ### Wait for Analysis
        
        Waits for the analysis pipeline to complete before generating recommendations.
        
        **Dependency**: analysis_pipeline.analysis_health_check
        **Timeout**: 10 minutes
        """,
    )

    # Signal generation task group
    with TaskGroup(
        "signal_generation", tooltip="Trading signal generation"
    ) as signal_generation:
        signals_task = PythonOperator(
            task_id="generate_trading_signals",
            python_callable=generate_trading_signals,
            doc_md="""
            ### Generate Trading Signals
            
            Generates trading signals based on comprehensive analysis:
            - Buy/Sell/Hold recommendations
            - Confidence scoring
            - Entry/exit point calculations
            - Risk-reward analysis
            """,
        )

    # Position sizing task group
    with TaskGroup(
        "position_sizing", tooltip="Position size calculation"
    ) as position_sizing:
        position_task = PythonOperator(
            task_id="calculate_position_sizes",
            python_callable=calculate_position_sizes,
            doc_md="""
            ### Calculate Position Sizes
            
            Calculates optimal position sizes:
            - Risk-based position sizing
            - Portfolio balance consideration
            - Maximum position limits
            - Risk management validation
            """,
        )

    # Portfolio optimization task group
    with TaskGroup(
        "portfolio_optimization", tooltip="Portfolio optimization"
    ) as portfolio_optimization:
        portfolio_task = PythonOperator(
            task_id="generate_portfolio_recommendations",
            python_callable=generate_portfolio_recommendations,
            doc_md="""
            ### Portfolio Optimization
            
            Generates portfolio-level recommendations:
            - Asset allocation optimization
            - Diversification analysis
            - Rebalancing recommendations
            - Expected return calculations
            """,
        )

    # Risk management task group
    with TaskGroup(
        "risk_management", tooltip="Risk management filters"
    ) as risk_management:
        risk_filter_task = PythonOperator(
            task_id="apply_risk_filters",
            python_callable=apply_risk_filters,
            doc_md="""
            ### Apply Risk Filters
            
            Applies comprehensive risk filters:
            - Position size validation
            - Portfolio risk limits
            - Market condition filters
            - Final recommendation validation
            """,
        )

    # Report generation task
    report_task = PythonOperator(
        task_id="generate_recommendation_report",
        python_callable=generate_recommendation_report,
        doc_md="""
        ### Generate Report
        
        Creates comprehensive recommendation report:
        - Executive summary
        - Detailed signal analysis
        - Risk assessment
        - Actionable recommendations
        - Performance projections
        """,
    )

    # Storage and notification task
    storage_task = PythonOperator(
        task_id="store_recommendations",
        python_callable=store_recommendations,
        doc_md="""
        ### Store & Notify
        
        Stores recommendations and sends notifications:
        - Database storage
        - User notifications
        - Tracking updates
        - Audit logging
        """,
    )

    # Final health check
    recommendation_health_check = BashOperator(
        task_id="recommendation_health_check",
        bash_command="""
        echo "Recommendation pipeline completed successfully"
        echo "Execution date: {{ ds }}"
        echo "Storage result: {{ task_instance.xcom_pull(task_ids='store_recommendations') }}"
        """,
        doc_md="""
        ### Recommendation Health Check
        
        Final verification of recommendation pipeline:
        - Confirms successful completion
        - Logs recommendation metrics
        - Validates data integrity
        """,
    )

    # Define task dependencies
    (
        wait_for_analysis
        >> signal_generation
        >> position_sizing
        >> portfolio_optimization
        >> risk_management
        >> report_task
        >> storage_task
        >> recommendation_health_check
    )
