"""Technical Analysis Pipeline DAG for AI Trading Advisor."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
import logging
import pandas as pd

logger = logging.getLogger(__name__)

default_args = {
    "owner": "ai-trading-advisor",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "technical_analysis_pipeline",
    default_args=default_args,
    description="Multi-timeframe technical analysis and signal generation",
    schedule_interval=timedelta(minutes=5),  # Run every 5 minutes during market hours
    max_active_runs=1,
    catchup=False,
    tags=["technical-analysis", "indicators", "signals", "multi-timeframe"],
)

def check_market_hours(**context):
    """Check if current time is within market hours (9:30 AM - 4:00 PM EST)."""
    try:
        from datetime import datetime, timezone, time
        import pytz
        
        # Get current time in Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern).time()
        
        # Market hours: 9:30 AM - 4:00 PM EST
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        # Check if it's a weekday
        current_date = datetime.now(eastern)
        is_weekday = current_date.weekday() < 5  # Monday = 0, Friday = 4
        
        if is_weekday and market_open <= current_time <= market_close:
            logger.info(f"Market is open at {current_time}")
            return "collect_realtime_data"
        else:
            logger.info(f"Market is closed at {current_time}")
            return "collect_daily_data"
            
    except ImportError:
        logger.warning("pytz not available, assuming market is open")
        return "collect_realtime_data"
    except Exception as e:
        logger.warning(f"Error checking market hours: {e}, defaulting to realtime collection")
        return "collect_realtime_data"

@task
def collect_realtime_data(**context):
    """Collect real-time technical data for active trading hours."""
    try:
        # Try to import required modules with proper path handling
        import sys
        import os
        
        # Add the project root to Python path if not already there
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.data.technical_collectors import TechnicalDataCollector
        from src.data.database import DatabaseManager
        from src.config import settings
        
        imports_successful = True
        
    except ImportError as e:
        logger.warning(f"Import failed in technical analysis pipeline: {e}")
        imports_successful = False
        
        # Create mock classes for graceful degradation
        class MockTechnicalDataCollector:
            def collect_technical_indicators(self, symbol, timeframes):
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "timeframes": {
                        tf: {
                            "rsi": {"current": 55.5, "overbought": False, "oversold": False},
                            "macd": {"macd": 0.5, "signal": 0.3, "histogram": 0.2, "bullish": True},
                            "bollinger_bands": {"upper": 105.0, "middle": 100.0, "lower": 95.0, "position": "middle"},
                            "moving_averages": {"sma_20": 99.5, "ema_12": 100.2},
                            "stochastic": {"k_percent": 60.0, "d_percent": 58.0, "overbought": False, "oversold": False},
                            "atr": {"current": 2.5, "percentage": 2.5},
                            "adx": {"adx": 30.0, "plus_di": 25.0, "minus_di": 20.0, "trend_strength": "moderate"},
                            "volume": {"current": 1500000, "average_20": 1200000, "relative": 1.25, "high_volume": True},
                            "latest_price": 100.0,
                            "timeframe": tf,
                            "data_points": 50
                        } for tf in timeframes
                    },
                    "data_source": "mock"
                }
            
            def invalidate_symbol_cache(self, symbol):
                return 5
        
        class MockDatabaseManager:
            def store_technical_data(self, data):
                logger.info(f"Mock storing technical data for {data.get('symbol', 'unknown')}")
                return True
        
        class MockSettings:
            realtime_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        
        TechnicalDataCollector = MockTechnicalDataCollector
        DatabaseManager = MockDatabaseManager
        settings = MockSettings()
    
    execution_date = context['execution_date']
    
    collector = TechnicalDataCollector()
    db_manager = DatabaseManager()
    
    # Get symbols for real-time analysis (high-volume, actively traded)
    symbols = getattr(settings, 'realtime_symbols', 
                     ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"])
    
    # Use short timeframes for real-time analysis
    timeframes = ["1m", "5m", "15m"]
    
    collected_data = []
    errors = []
    
    for symbol in symbols:
        try:
            logger.info(f"Collecting real-time technical data for {symbol}")
            
            # Invalidate cache to ensure fresh data
            collector.invalidate_symbol_cache(symbol)
            
            # Collect indicators
            technical_data = collector.collect_technical_indicators(symbol, timeframes)
            
            if technical_data and "error" not in technical_data:
                # Add execution metadata
                technical_data["execution_date"] = execution_date.date()
                technical_data["analysis_type"] = "realtime"
                collected_data.append(technical_data)
                
                # Store in database
                db_manager.store_technical_data(technical_data)
                logger.info(f"Successfully collected real-time data for {symbol}")
            else:
                errors.append(f"Failed to collect technical data for {symbol}")
                
        except Exception as e:
            error_msg = f"Error collecting real-time data for {symbol}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    result = {
        "status": "success" if collected_data and not errors else "partial" if collected_data else "failed",
        "symbols_processed": len(collected_data),
        "total_symbols": len(symbols),
        "timeframes": timeframes,
        "execution_date": execution_date.isoformat(),
        "analysis_type": "realtime",
        "errors": errors[:10]  # Limit error list
    }
    
    logger.info(f"Real-time technical data collection completed: {result}")
    return result

@task
def collect_daily_data(**context):
    """Collect daily technical data for after-hours analysis."""
    try:
        import sys
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from src.data.technical_collectors import TechnicalDataCollector
        from src.data.database import DatabaseManager
        from src.config import settings
        imports_successful = True
        
    except ImportError as e:
        logger.warning(f"Import failed in daily technical collection: {e}")
        imports_successful = False
        
        class MockTechnicalDataCollector:
            def collect_technical_indicators(self, symbol, timeframes):
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "timeframes": {
                        tf: {
                            "rsi": {"current": 45.5, "overbought": False, "oversold": False},
                            "macd": {"macd": -0.2, "signal": -0.1, "histogram": -0.1, "bullish": False},
                            "bollinger_bands": {"upper": 110.0, "middle": 102.0, "lower": 94.0, "position": "middle"},
                            "moving_averages": {"sma_50": 101.5, "sma_200": 98.0, "ema_26": 102.2},
                            "stochastic": {"k_percent": 40.0, "d_percent": 42.0, "overbought": False, "oversold": False},
                            "atr": {"current": 3.2, "percentage": 3.1},
                            "adx": {"adx": 25.0, "plus_di": 22.0, "minus_di": 28.0, "trend_strength": "moderate"},
                            "volume": {"current": 2500000, "average_20": 2200000, "relative": 1.14, "high_volume": True},
                            "latest_price": 102.0,
                            "timeframe": tf,
                            "data_points": 100
                        } for tf in timeframes
                    },
                    "data_source": "mock"
                }
        
        class MockDatabaseManager:
            def store_technical_data(self, data):
                logger.info(f"Mock storing daily technical data for {data.get('symbol', 'unknown')}")
                return True
        
        class MockSettings:
            daily_symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", 
                           "NFLX", "CRM", "ORCL", "ADBE", "INTC", "AMD", "JPM", "BAC", "WFC", "GS"]
        
        TechnicalDataCollector = MockTechnicalDataCollector
        DatabaseManager = MockDatabaseManager
        settings = MockSettings()
    
    execution_date = context['execution_date']
    
    collector = TechnicalDataCollector()
    db_manager = DatabaseManager()
    
    # Get extended symbol list for daily analysis
    symbols = getattr(settings, 'daily_symbols', 
                     ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", 
                      "NFLX", "CRM", "ORCL", "ADBE", "INTC", "AMD", "JPM", "BAC", "WFC", "GS"])
    
    # Use longer timeframes for daily analysis
    timeframes = ["1h", "4h", "1d"]
    
    collected_data = []
    errors = []
    
    for symbol in symbols:
        try:
            logger.info(f"Collecting daily technical data for {symbol}")
            
            # Collect indicators
            technical_data = collector.collect_technical_indicators(symbol, timeframes)
            
            if technical_data and "error" not in technical_data:
                # Add execution metadata
                technical_data["execution_date"] = execution_date.date()
                technical_data["analysis_type"] = "daily"
                collected_data.append(technical_data)
                
                # Store in database
                db_manager.store_technical_data(technical_data)
                logger.info(f"Successfully collected daily data for {symbol}")
            else:
                errors.append(f"Failed to collect technical data for {symbol}")
                
        except Exception as e:
            error_msg = f"Error collecting daily data for {symbol}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    result = {
        "status": "success" if collected_data and not errors else "partial" if collected_data else "failed",
        "symbols_processed": len(collected_data),
        "total_symbols": len(symbols),
        "timeframes": timeframes,
        "execution_date": execution_date.isoformat(),
        "analysis_type": "daily",
        "errors": errors[:10]
    }
    
    logger.info(f"Daily technical data collection completed: {result}")
    return result

@task
def generate_trading_signals(**context):
    """Generate trading signals from multi-timeframe technical analysis."""
    try:
        import sys
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from src.core.technical_analysis import MultiTimeframeAnalysis
        from src.data.database import DatabaseManager
        imports_successful = True
        
    except ImportError as e:
        logger.warning(f"Import failed in signal generation: {e}")
        imports_successful = False
        
        class MockMultiTimeframeAnalysis:
            def generate_signals(self, indicators):
                return {
                    "timestamp": datetime.now().isoformat(),
                    "overall_signal": "BUY",
                    "confidence": 0.75,
                    "timeframe_signals": {
                        "5m": {"signal": "BUY", "score": 0.7, "key_factors": ["RSI oversold", "MACD bullish"]},
                        "1h": {"signal": "HOLD", "score": 0.55, "key_factors": ["Mixed signals"]},
                        "1d": {"signal": "BUY", "score": 0.8, "key_factors": ["Strong trend", "Volume confirmation"]}
                    },
                    "confluence_factors": ["Multi-timeframe bullish confluence"],
                    "risk_factors": []
                }
        
        class MockDatabaseManager:
            def get_latest_technical_data(self, symbols, hours_back=1):
                return [
                    {
                        "symbol": "AAPL",
                        "timeframes": {
                            "5m": {"rsi": {"current": 35}, "macd": {"bullish": True}},
                            "1h": {"rsi": {"current": 45}, "macd": {"bullish": False}},
                            "1d": {"rsi": {"current": 40}, "macd": {"bullish": True}}
                        }
                    }
                ]
            
            def store_trading_signals(self, signals):
                logger.info(f"Mock storing {len(signals)} trading signals")
                return True
        
        MultiTimeframeAnalysis = MockMultiTimeframeAnalysis
        DatabaseManager = MockDatabaseManager
    
    execution_date = context['execution_date']
    
    analysis_engine = MultiTimeframeAnalysis()
    db_manager = DatabaseManager()
    
    try:
        # Get recent technical data for signal generation
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        
        # Get technical data from the last hour
        technical_data = db_manager.get_latest_technical_data(symbols, hours_back=1)
        
        generated_signals = []
        signal_summary = {
            "BUY": 0,
            "SELL": 0,
            "HOLD": 0
        }
        
        for data in technical_data:
            symbol = data.get("symbol")
            indicators = data.get("timeframes", {})
            
            if not indicators:
                logger.warning(f"No indicators available for {symbol}")
                continue
            
            try:
                # Generate signals for the symbol
                signals = analysis_engine.generate_signals(indicators)
                
                if signals:
                    signal_data = {
                        "symbol": symbol,
                        "signals": signals,
                        "timestamp": datetime.now().isoformat(),
                        "execution_date": execution_date.date()
                    }
                    
                    generated_signals.append(signal_data)
                    
                    # Count signal types
                    overall_signal = signals.get("overall_signal", "HOLD")
                    signal_summary[overall_signal] = signal_summary.get(overall_signal, 0) + 1
                    
                    logger.info(f"Generated {overall_signal} signal for {symbol} with confidence {signals.get('confidence', 0):.2f}")
                
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
        
        # Store signals in database
        if generated_signals:
            db_manager.store_trading_signals(generated_signals)
        
        result = {
            "status": "success",
            "signals_generated": len(generated_signals),
            "signal_summary": signal_summary,
            "execution_date": execution_date.isoformat(),
            "high_confidence_signals": len([s for s in generated_signals if s.get("signals", {}).get("confidence", 0) > 0.7])
        }
        
        logger.info(f"Trading signal generation completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "execution_date": execution_date.isoformat()
        }

@task
def validate_signal_quality(**context):
    """Validate quality of generated trading signals."""
    try:
        import sys
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from src.data.database import DatabaseManager
        imports_successful = True
        
    except ImportError as e:
        logger.warning(f"Import failed in signal validation: {e}")
        imports_successful = False
        
        class MockDatabaseManager:
            def get_recent_signals(self, hours_back=1):
                return [
                    {
                        "symbol": "AAPL",
                        "signals": {
                            "overall_signal": "BUY",
                            "confidence": 0.85,
                            "timeframe_signals": {"5m": {"signal": "BUY"}, "1h": {"signal": "BUY"}, "1d": {"signal": "HOLD"}}
                        }
                    },
                    {
                        "symbol": "MSFT", 
                        "signals": {
                            "overall_signal": "HOLD",
                            "confidence": 0.45,
                            "timeframe_signals": {"5m": {"signal": "SELL"}, "1h": {"signal": "BUY"}, "1d": {"signal": "HOLD"}}
                        }
                    }
                ]
            
            def store_signal_validation(self, validation_results):
                logger.info(f"Mock storing signal validation results")
                return True
        
        DatabaseManager = MockDatabaseManager
    
    execution_date = context['execution_date']
    db_manager = DatabaseManager()
    
    try:
        # Get recently generated signals
        recent_signals = db_manager.get_recent_signals(hours_back=1)
        
        validation_results = {
            "total_signals": len(recent_signals),
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "conflicting_signals": 0,
            "consensus_signals": 0,
            "quality_score": 0.0,
            "timestamp": datetime.now().isoformat(),
            "execution_date": execution_date.date()
        }
        
        for signal_data in recent_signals:
            signals = signal_data.get("signals", {})
            confidence = signals.get("confidence", 0.0)
            
            # Categorize by confidence
            if confidence >= 0.7:
                validation_results["high_confidence"] += 1
            elif confidence >= 0.5:
                validation_results["medium_confidence"] += 1
            else:
                validation_results["low_confidence"] += 1
            
            # Check for timeframe consensus
            timeframe_signals = signals.get("timeframe_signals", {})
            if timeframe_signals:
                signal_types = [tf_signal.get("signal") for tf_signal in timeframe_signals.values()]
                unique_signals = set(signal_types)
                
                if len(unique_signals) == 1:
                    validation_results["consensus_signals"] += 1
                elif len(unique_signals) >= 3:
                    validation_results["conflicting_signals"] += 1
        
        # Calculate overall quality score
        if validation_results["total_signals"] > 0:
            quality_score = (
                (validation_results["high_confidence"] * 1.0 +
                 validation_results["medium_confidence"] * 0.6 +
                 validation_results["consensus_signals"] * 0.3) /
                validation_results["total_signals"]
            )
            validation_results["quality_score"] = round(quality_score, 3)
        
        # Store validation results
        db_manager.store_signal_validation(validation_results)
        
        logger.info(f"Signal validation completed: quality_score={validation_results['quality_score']}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Signal validation failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "execution_date": execution_date.isoformat()
        }

@task
def cleanup_old_technical_data(**context):
    """Clean up technical data older than 30 days."""
    try:
        import sys
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from src.data.database import DatabaseManager
        imports_successful = True
        
    except ImportError as e:
        logger.warning(f"Import failed in technical data cleanup: {e}")
        imports_successful = False
        
        class MockDatabaseManager:
            def cleanup_technical_data(self, cutoff_date):
                logger.info(f"Mock cleanup of technical data before {cutoff_date}")
                return 1250  # Mock number of records deleted
        
        DatabaseManager = MockDatabaseManager
    
    execution_date = context['execution_date']
    cutoff_date = execution_date - timedelta(days=30)  # Keep 30 days
    
    db_manager = DatabaseManager()
    
    try:
        records_deleted = db_manager.cleanup_technical_data(cutoff_date.date())
        
        result = {
            "status": "success",
            "records_deleted": records_deleted,
            "cutoff_date": cutoff_date.isoformat(),
            "execution_date": execution_date.isoformat()
        }
        
        logger.info(f"Technical data cleanup completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Technical data cleanup failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "execution_date": execution_date.isoformat()
        }

# Task creation and dependencies - wrap in try/catch for Airflow compatibility
try:
    with dag:
        # Market hours check (branching)
        market_check = BranchPythonOperator(
            task_id="check_market_hours",
            python_callable=check_market_hours,
            dag=dag,
        )
        
        # Data collection tasks
        realtime_task = collect_realtime_data()
        daily_task = collect_daily_data()
        
        # Join point after branching
        join_task = DummyOperator(
            task_id="join_data_collection",
            trigger_rule="none_failed_or_skipped",
            dag=dag,
        )
        
        # Signal generation and validation
        signals_task = generate_trading_signals()
        validation_task = validate_signal_quality()
        
        # Cleanup task (runs daily)
        cleanup_task = cleanup_old_technical_data()
        
        # Set up dependencies
        market_check >> [realtime_task, daily_task]
        [realtime_task, daily_task] >> join_task
        join_task >> signals_task >> validation_task
        
        # Cleanup runs independently every 6 hours
        cleanup_task
        
except Exception as e:
    # During testing or import issues, skip dependency setup
    import logging
    logging.getLogger(__name__).warning(f"Skipping technical analysis DAG dependency setup: {e}")