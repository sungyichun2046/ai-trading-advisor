"""Real-time volatility monitoring pipeline with dynamic triggers."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.context import Context
import logging

logger = logging.getLogger(__name__)

default_args = {
    "owner": "ai-trading-advisor",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "volatility_monitoring",
    default_args=default_args,
    description="Real-time volatility monitoring with emergency analysis triggers",
    schedule_interval=timedelta(minutes=15),  # Check every 15 minutes during market hours
    max_active_runs=3,
    catchup=False,
    start_date=datetime(2024, 1, 1),  # Explicitly set start_date
    tags=["real-time", "volatility", "monitoring", "alerts"],
)

class VolatilitySensor(BaseSensorOperator):
    """Custom sensor for monitoring market volatility conditions."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timeout = 60 * 15  # 15 minutes
        self.poke_interval = 60 * 5  # Check every 5 minutes
        
    def poke(self, context: Context) -> bool:
        """Check volatility conditions."""
        try:
            # Try to import VolatilityMonitor with fallback
            try:
                from src.data.collectors import VolatilityMonitor
                monitor = VolatilityMonitor()
                volatility_data = monitor.check_market_volatility()
            except ImportError as e:
                logger.warning(f"VolatilityMonitor import failed: {e}, using mock data")
                # Mock volatility data for development/testing
                volatility_data = {
                    "status": "success",
                    "volatility_level": "NORMAL",
                    "vix_current": 18.5,
                    "triggers": [],
                    "alerts": [],
                    "data_source": "mock",
                    "timestamp": datetime.now().isoformat()
                }
            
            if volatility_data.get("status") != "success":
                logger.warning("Volatility check failed, will retry")
                return False
            
            # Store volatility data in XCom for downstream tasks
            context['task_instance'].xcom_push(key='volatility_data', value=volatility_data)
            
            # Check if any triggers are needed
            triggers = volatility_data.get("triggers", [])
            alerts = volatility_data.get("alerts", [])
            
            logger.info(f"Volatility check completed: {len(alerts)} alerts, {len(triggers)} triggers")
            
            # Always return True to continue the DAG - triggers are handled downstream
            return True
            
        except Exception as e:
            logger.error(f"Volatility sensor failed: {e}")
            return False

@task
def evaluate_volatility_conditions(**context):
    """Evaluate volatility conditions and determine actions."""
    
    # Get volatility data from sensor
    volatility_data = context['task_instance'].xcom_pull(
        task_ids='volatility_sensor',
        key='volatility_data'
    )
    
    if not volatility_data:
        logger.error("No volatility data available from sensor")
        return {"status": "failed", "reason": "no_data"}
    
    execution_date = context['execution_date']
    triggers = volatility_data.get("triggers", [])
    alerts = volatility_data.get("alerts", [])
    volatility_level = volatility_data.get("volatility_level", "UNKNOWN")
    
    # Store volatility event in database
    try:
        from src.data.database import DatabaseManager
        db_manager = DatabaseManager()
        volatility_event = {
            "timestamp": execution_date.isoformat(),
            "volatility_level": volatility_level,
            "vix_current": volatility_data.get("vix_current"),
            "alerts": alerts,
            "triggers": triggers,
            "data_source": volatility_data.get("data_source", "unknown"),
            "execution_date": execution_date.date()
        }
        
        db_manager.store_volatility_event(volatility_event)
        logger.info(f"Volatility event stored: {volatility_level}")
        
    except Exception as e:
        logger.error(f"Failed to store volatility event: {e}")
    
    # Prepare trigger decisions
    trigger_decisions = {
        "emergency_analysis": "emergency_analysis" in triggers,
        "enhanced_monitoring": "enhanced_monitoring" in triggers,
        "market_stress_analysis": "market_stress_analysis" in triggers,
        "volatility_level": volatility_level,
        "alert_count": len(alerts),
        "execution_date": execution_date.isoformat()
    }
    
    logger.info(f"Volatility evaluation completed: {trigger_decisions}")
    return trigger_decisions

@task
def check_market_hours(**context):
    """Check if market is currently open for trading."""
    try:
        import pandas_market_calendars as mcal
    except ImportError:
        logger.warning("pandas_market_calendars not available, assuming market is open")
        return {"market_open": True, "reason": "calendar_unavailable"}
    
    try:
        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        now = datetime.now()
        
        # Check if market is open right now
        schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
        
        if schedule.empty:
            logger.info("Market is closed today")
            return {"market_open": False, "reason": "market_closed"}
        
        # Check if current time is within market hours
        market_open = schedule.iloc[0]['market_open'].tz_localize(None)
        market_close = schedule.iloc[0]['market_close'].tz_localize(None)
        
        is_open = market_open <= now <= market_close
        
        logger.info(f"Market hours check: {'OPEN' if is_open else 'CLOSED'} at {now}")
        
        return {
            "market_open": is_open,
            "market_open_time": market_open.isoformat(),
            "market_close_time": market_close.isoformat(),
            "current_time": now.isoformat()
        }
        
    except Exception as e:
        logger.warning(f"Market hours check failed: {e}, assuming market is open")
        # Default to assuming market is open to avoid missing critical alerts
        return {"market_open": True, "reason": "check_failed"}

@task.branch
def decide_actions(**context):
    """Decide which actions to take based on volatility conditions and market hours."""
    
    # Get evaluation results
    volatility_decisions = context['task_instance'].xcom_pull(task_ids='evaluate_volatility_conditions')
    market_hours = context['task_instance'].xcom_pull(task_ids='check_market_hours')
    
    if not volatility_decisions:
        logger.error("No volatility decisions available")
        return 'no_action'
    
    # Check market hours
    market_open = market_hours.get("market_open", True) if market_hours else True
    
    if not market_open:
        logger.info("Market is closed, skipping volatility triggers")
        return 'log_volatility_event'
    
    # Determine actions based on volatility level and triggers
    volatility_level = volatility_decisions.get("volatility_level")
    emergency_analysis = volatility_decisions.get("emergency_analysis", False)
    enhanced_monitoring = volatility_decisions.get("enhanced_monitoring", False)
    market_stress_analysis = volatility_decisions.get("market_stress_analysis", False)
    
    actions = []
    
    if emergency_analysis and volatility_level in ["EXTREME", "HIGH"]:
        actions.append("trigger_emergency_analysis")
        logger.warning(f"EMERGENCY: Triggering emergency analysis due to {volatility_level} volatility")
    
    if enhanced_monitoring:
        actions.append("trigger_enhanced_monitoring")
        logger.info("Triggering enhanced monitoring due to elevated volatility")
    
    if market_stress_analysis:
        actions.append("trigger_market_stress_analysis")
        logger.info("Triggering market stress analysis")
    
    # Always log the event
    actions.append("log_volatility_event")
    
    if not actions or actions == ["log_volatility_event"]:
        logger.info("No special actions needed, normal volatility conditions")
        return 'log_volatility_event'
    
    # Return first critical action (emergency has highest priority)
    if "trigger_emergency_analysis" in actions:
        return 'trigger_emergency_analysis'
    elif "trigger_market_stress_analysis" in actions:
        return 'trigger_market_stress_analysis'
    elif "trigger_enhanced_monitoring" in actions:
        return 'trigger_enhanced_monitoring'
    else:
        return 'log_volatility_event'

@task
def log_volatility_event(**context):
    """Log volatility monitoring event."""
    execution_date = context['execution_date']
    
    volatility_decisions = context['task_instance'].xcom_pull(task_ids='evaluate_volatility_conditions')
    market_hours = context['task_instance'].xcom_pull(task_ids='check_market_hours')
    
    log_entry = {
        "timestamp": execution_date.isoformat(),
        "volatility_level": volatility_decisions.get("volatility_level") if volatility_decisions else "UNKNOWN",
        "market_open": market_hours.get("market_open") if market_hours else "UNKNOWN",
        "alert_count": volatility_decisions.get("alert_count", 0) if volatility_decisions else 0,
        "action": "logged"
    }
    
    logger.info(f"Volatility monitoring event logged: {log_entry}")
    return log_entry

@task
def no_action(**context):
    """No action needed - placeholder task."""
    logger.info("No volatility actions required")
    return {"status": "no_action", "timestamp": context['execution_date'].isoformat()}

# Define sensor and main tasks
volatility_sensor = VolatilitySensor(
    task_id='volatility_sensor',
    dag=dag
)

evaluate_conditions = evaluate_volatility_conditions.override(task_id='evaluate_volatility_conditions')()
check_hours = check_market_hours.override(task_id='check_market_hours')()
decide_actions_task = decide_actions.override(task_id='decide_actions')()
log_event = log_volatility_event.override(task_id='log_volatility_event')()
no_action_task = no_action.override(task_id='no_action')()

# Emergency analysis trigger (triggers analysis_pipeline DAG)
trigger_emergency_analysis = TriggerDagRunOperator(
    task_id='trigger_emergency_analysis',
    trigger_dag_id='analysis_pipeline',
    conf={
        "emergency_mode": True,
        "volatility_trigger": True,
        "priority": "CRITICAL"
    },
    dag=dag
)

# Enhanced monitoring trigger (triggers data_collection_pipeline with higher frequency)
trigger_enhanced_monitoring = TriggerDagRunOperator(
    task_id='trigger_enhanced_monitoring',
    trigger_dag_id='data_collection_pipeline',
    conf={
        "enhanced_mode": True,
        "volatility_trigger": True,
        "priority": "HIGH"
    },
    dag=dag
)

# Market stress analysis trigger (triggers recommendation_pipeline with stress testing)
trigger_market_stress_analysis = TriggerDagRunOperator(
    task_id='trigger_market_stress_analysis',
    trigger_dag_id='recommendation_pipeline',
    conf={
        "stress_test_mode": True,
        "volatility_trigger": True,
        "priority": "HIGH"
    },
    dag=dag
)

# Set up dependencies - wrap in try/catch for Airflow compatibility
try:
    # Set up dependencies
    volatility_sensor >> [evaluate_conditions, check_hours]
    [evaluate_conditions, check_hours] >> decide_actions_task

    # Branch to different actions
    decide_actions_task >> [
        trigger_emergency_analysis,
        trigger_enhanced_monitoring, 
        trigger_market_stress_analysis,
        log_event,
        no_action_task
    ]

    # All trigger paths should end with logging
    trigger_emergency_analysis >> log_event
    trigger_enhanced_monitoring >> log_event
    trigger_market_stress_analysis >> log_event
except Exception as e:
    # During testing or import issues, skip dependency setup
    import logging
    logging.getLogger(__name__).warning(f"Skipping DAG dependency setup: {e}")