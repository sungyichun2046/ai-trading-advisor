"""Sector Correlation Pipeline DAG for AI Trading Advisor.

This DAG provides daily sector analysis and correlation monitoring including:
- ETF sector performance tracking and ranking
- Cross-asset correlation calculations and alerts
- Sector rotation signal generation
- Risk regime detection and portfolio implications
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
import logging
import pandas as pd
import numpy as np

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
    "sector_correlation_pipeline",
    default_args=default_args,
    description="Daily sector analysis and correlation monitoring with rotation signals",
    schedule_interval=timedelta(days=1),  # Run daily
    max_active_runs=1,
    catchup=False,
    tags=["sector-analysis", "correlation", "rotation", "cross-asset", "monitoring"],
)

def check_market_session(**context):
    """Check market session and determine analysis scope."""
    try:
        from datetime import datetime, timezone, time
        import pytz
        
        # Get current time in Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern).time()
        current_date = datetime.now(eastern)
        
        # Market hours: 9:30 AM - 4:00 PM EST
        market_open = time(9, 30)
        market_close = time(16, 0)
        is_weekday = current_date.weekday() < 5
        
        # After market close or weekend
        if not is_weekday or current_time > market_close:
            logger.info(f"Post-market or weekend analysis at {current_time}")
            return "collect_comprehensive_data"
        elif is_weekday and market_open <= current_time <= market_close:
            logger.info(f"Intraday analysis during market hours at {current_time}")
            return "collect_intraday_data"
        else:
            logger.info(f"Pre-market analysis at {current_time}")
            return "collect_premarket_data"
            
    except ImportError:
        logger.warning("pytz not available, defaulting to comprehensive analysis")
        return "collect_comprehensive_data"
    except Exception as e:
        logger.warning(f"Error checking market session: {e}, defaulting to comprehensive analysis")
        return "collect_comprehensive_data"

@task
def collect_comprehensive_data(**context):
    """Collect comprehensive sector and asset data for post-market analysis."""
    try:
        # Import handling with graceful degradation
        import sys
        import os
        
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.data.collectors import MarketDataCollector
        from src.data.database import DatabaseManager
        from src.config import settings
        
        imports_successful = True
        
    except ImportError as e:
        logger.warning(f"Import failed in sector correlation pipeline: {e}")
        imports_successful = False
    
    if imports_successful:
        try:
            # Collect data for comprehensive analysis
            collector = MarketDataCollector()
            
            # Define comprehensive symbol universe
            sector_etfs = [
                # SPDR Sector ETFs
                'XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLC',
                # Vanguard Sector ETFs
                'VGT', 'VHT', 'VFH', 'VCR', 'VDC', 'VDE', 'VIS', 'VAW', 'VNQ', 'VPU', 'VOX',
                # Broad Market ETFs
                'SPY', 'QQQ', 'IWM', 'VTI', 'VOO',
                # International
                'EFA', 'EEM', 'VEA', 'VWO',
                # Bonds
                'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'AGG', 'BND',
                # Commodities
                'GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC',
                # REITs
                'VNQ', 'SCHH', 'RWR',
                # Currencies
                'UUP', 'FXE', 'FXY'
            ]
            
            collected_data = {}
            for symbol in sector_etfs:
                try:
                    # Get different timeframes for comprehensive analysis
                    daily_data = collector.collect_historical_data(symbol, "1d", lookback=252)  # 1 year
                    weekly_data = collector.collect_historical_data(symbol, "1w", lookback=104)  # 2 years
                    
                    collected_data[symbol] = {
                        "daily": daily_data,
                        "weekly": weekly_data,
                        "timestamp": datetime.now().isoformat(),
                        "data_quality": "comprehensive"
                    }
                    
                    logger.info(f"Collected comprehensive data for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {e}")
                    collected_data[symbol] = {"error": str(e)}
            
            # Store in XCom for next tasks
            context['task_instance'].xcom_push(key='market_data', value=collected_data)
            
            return {
                "status": "success",
                "symbols_processed": len(collected_data),
                "mode": "comprehensive",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive data collection: {e}")
            return {"status": "error", "error": str(e)}
    
    else:
        # Mock comprehensive data for when imports fail
        mock_data = {}
        sector_etfs = [
            'XLK', 'XLV', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLC',
            'SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VNQ'
        ]
        
        for symbol in sector_etfs:
            # Generate realistic mock OHLCV data
            dates = pd.date_range(end=datetime.now(), periods=252, freq='1D')
            base_price = np.random.uniform(50, 300)
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
            prices = base_price * (1 + returns).cumprod()
            
            daily_data = pd.DataFrame({
                'Open': prices * np.random.uniform(0.998, 1.002, len(dates)),
                'High': prices * np.random.uniform(1.005, 1.015, len(dates)),
                'Low': prices * np.random.uniform(0.985, 0.995, len(dates)),
                'Close': prices,
                'Volume': np.random.randint(1000000, 50000000, len(dates))
            }, index=dates)
            
            # Weekly data (every 7 days)
            weekly_data = daily_data[::7].copy()
            weekly_data['Volume'] = weekly_data['Volume'] * 5  # Aggregate volume
            
            mock_data[symbol] = {
                "daily": daily_data,
                "weekly": weekly_data,
                "timestamp": datetime.now().isoformat(),
                "data_quality": "mock_comprehensive"
            }
        
        context['task_instance'].xcom_push(key='market_data', value=mock_data)
        
        return {
            "status": "success_mock",
            "symbols_processed": len(mock_data),
            "mode": "comprehensive_mock",
            "timestamp": datetime.now().isoformat()
        }

@task
def collect_intraday_data(**context):
    """Collect intraday data for real-time analysis during market hours."""
    try:
        import sys
        import os
        
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.data.collectors import MarketDataCollector
        
        collector = MarketDataCollector()
        
        # Focus on most liquid sector ETFs for intraday analysis
        intraday_symbols = [
            'SPY', 'QQQ', 'IWM', 'XLK', 'XLV', 'XLF', 'XLY', 'XLE', 'TLT', 'GLD'
        ]
        
        collected_data = {}
        for symbol in intraday_symbols:
            try:
                # Get shorter-term data for intraday analysis
                hourly_data = collector.collect_historical_data(symbol, "1h", lookback=168)  # 1 week
                daily_data = collector.collect_historical_data(symbol, "1d", lookback=63)   # 3 months
                
                collected_data[symbol] = {
                    "hourly": hourly_data,
                    "daily": daily_data,
                    "timestamp": datetime.now().isoformat(),
                    "data_quality": "intraday"
                }
                
            except Exception as e:
                logger.error(f"Error collecting intraday data for {symbol}: {e}")
                collected_data[symbol] = {"error": str(e)}
        
        context['task_instance'].xcom_push(key='market_data', value=collected_data)
        
        return {
            "status": "success",
            "symbols_processed": len(collected_data),
            "mode": "intraday",
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        # Mock intraday data
        mock_data = {}
        intraday_symbols = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLV', 'XLF', 'TLT', 'GLD']
        
        for symbol in intraday_symbols:
            dates_hourly = pd.date_range(end=datetime.now(), periods=168, freq='1H')
            dates_daily = pd.date_range(end=datetime.now(), periods=63, freq='1D')
            
            base_price = np.random.uniform(100, 400)
            
            # Hourly data
            hourly_returns = np.random.normal(0, 0.005, len(dates_hourly))
            hourly_prices = base_price * (1 + hourly_returns).cumprod()
            
            hourly_data = pd.DataFrame({
                'Open': hourly_prices * 0.999,
                'High': hourly_prices * 1.002,
                'Low': hourly_prices * 0.998,
                'Close': hourly_prices,
                'Volume': np.random.randint(100000, 5000000, len(dates_hourly))
            }, index=dates_hourly)
            
            # Daily data
            daily_returns = np.random.normal(0.0003, 0.015, len(dates_daily))
            daily_prices = base_price * (1 + daily_returns).cumprod()
            
            daily_data = pd.DataFrame({
                'Open': daily_prices * 0.999,
                'High': daily_prices * 1.01,
                'Low': daily_prices * 0.99,
                'Close': daily_prices,
                'Volume': np.random.randint(5000000, 100000000, len(dates_daily))
            }, index=dates_daily)
            
            mock_data[symbol] = {
                "hourly": hourly_data,
                "daily": daily_data,
                "timestamp": datetime.now().isoformat(),
                "data_quality": "mock_intraday"
            }
        
        context['task_instance'].xcom_push(key='market_data', value=mock_data)
        
        return {
            "status": "success_mock",
            "symbols_processed": len(mock_data),
            "mode": "intraday_mock",
            "timestamp": datetime.now().isoformat()
        }

@task
def collect_premarket_data(**context):
    """Collect pre-market data for gap analysis and overnight moves."""
    try:
        import sys
        import os
        
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.data.collectors import MarketDataCollector
        
        collector = MarketDataCollector()
        
        # Focus on key benchmarks and sector leaders for pre-market
        premarket_symbols = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'TLT', 'GLD', 'VNQ']
        
        collected_data = {}
        for symbol in premarket_symbols:
            try:
                # Get recent daily data for gap analysis
                daily_data = collector.collect_historical_data(symbol, "1d", lookback=126)  # 6 months
                
                collected_data[symbol] = {
                    "daily": daily_data,
                    "timestamp": datetime.now().isoformat(),
                    "data_quality": "premarket"
                }
                
            except Exception as e:
                logger.error(f"Error collecting premarket data for {symbol}: {e}")
                collected_data[symbol] = {"error": str(e)}
        
        context['task_instance'].xcom_push(key='market_data', value=collected_data)
        
        return {
            "status": "success",
            "symbols_processed": len(collected_data),
            "mode": "premarket",
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        # Mock premarket data
        mock_data = {}
        premarket_symbols = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'TLT', 'GLD', 'VNQ']
        
        for symbol in premarket_symbols:
            dates = pd.date_range(end=datetime.now(), periods=126, freq='1D')
            base_price = np.random.uniform(80, 350)
            returns = np.random.normal(0.0005, 0.018, len(dates))
            prices = base_price * (1 + returns).cumprod()
            
            daily_data = pd.DataFrame({
                'Open': prices * np.random.uniform(0.995, 1.005, len(dates)),
                'High': prices * np.random.uniform(1.008, 1.020, len(dates)),
                'Low': prices * np.random.uniform(0.980, 0.992, len(dates)),
                'Close': prices,
                'Volume': np.random.randint(10000000, 200000000, len(dates))
            }, index=dates)
            
            mock_data[symbol] = {
                "daily": daily_data,
                "timestamp": datetime.now().isoformat(),
                "data_quality": "mock_premarket"
            }
        
        context['task_instance'].xcom_push(key='market_data', value=mock_data)
        
        return {
            "status": "success_mock",
            "symbols_processed": len(mock_data),
            "mode": "premarket_mock",
            "timestamp": datetime.now().isoformat()
        }

@task
def analyze_sector_performance(**context):
    """Analyze sector performance and generate rankings."""
    try:
        import sys
        import os
        
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.core.sector_analysis import SectorPerformanceAnalyzer
        
        # Get data from previous task
        market_data = context['task_instance'].xcom_pull(key='market_data')
        
        if not market_data:
            logger.error("No market data available for sector analysis")
            return {"status": "error", "error": "No market data"}
        
        analyzer = SectorPerformanceAnalyzer()
        
        # Prepare price data for analysis
        price_data = {}
        benchmark_data = None
        
        for symbol, symbol_data in market_data.items():
            if "error" in symbol_data:
                continue
            
            # Use daily data if available, otherwise the longest timeframe
            if "daily" in symbol_data and isinstance(symbol_data["daily"], pd.DataFrame):
                price_data[symbol] = symbol_data["daily"]
                
                # Use SPY as benchmark if available
                if symbol == "SPY":
                    benchmark_data = symbol_data["daily"]
            elif "hourly" in symbol_data and isinstance(symbol_data["hourly"], pd.DataFrame):
                price_data[symbol] = symbol_data["hourly"]
        
        if not price_data:
            logger.error("No valid price data for sector analysis")
            return {"status": "error", "error": "No valid price data"}
        
        # Perform sector analysis
        sector_analysis = analyzer.calculate_sector_performance(price_data, benchmark_data)
        
        # Store results
        context['task_instance'].xcom_push(key='sector_analysis', value=sector_analysis)
        
        return {
            "status": "success",
            "sectors_analyzed": len(price_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError as e:
        logger.warning(f"Sector analysis imports failed: {e}, using mock analysis")
        
        # Mock sector analysis results
        market_data = context['task_instance'].xcom_pull(key='market_data') or {}
        mock_analysis = {
            "timestamp": datetime.now().isoformat(),
            "sector_performance": {},
            "sector_rankings": {
                "by_medium_term_return": [
                    {"symbol": "XLK", "sector": "Technology", "return": 0.08, "rank": 1},
                    {"symbol": "XLV", "sector": "Healthcare", "return": 0.05, "rank": 2},
                    {"symbol": "XLF", "sector": "Financial", "return": 0.03, "rank": 3},
                    {"symbol": "XLE", "sector": "Energy", "return": -0.02, "rank": 4},
                    {"symbol": "XLU", "sector": "Utilities", "return": -0.05, "rank": 5}
                ]
            },
            "performance_summary": {
                "total_sectors_analyzed": 5,
                "best_performing_sector": {"symbol": "XLK", "sector": "Technology", "return": 0.08},
                "worst_performing_sector": {"symbol": "XLU", "sector": "Utilities", "return": -0.05},
                "average_performance": {"mean_return": 0.02, "std_return": 0.05}
            },
            "rotation_candidates": {
                "rotation_into": [
                    {"symbol": "XLK", "sector": "Technology", "signal": "strong_buy", "momentum": 0.12}
                ],
                "rotation_out_of": [
                    {"symbol": "XLU", "sector": "Utilities", "signal": "sell", "momentum": -0.08}
                ]
            }
        }
        
        # Add mock performance for each symbol
        for symbol in market_data.keys():
            mock_analysis["sector_performance"][symbol] = {
                "symbol": symbol,
                "sector": "Mock Sector",
                "period_returns": {
                    "short_term": np.random.uniform(-0.05, 0.05),
                    "medium_term": np.random.uniform(-0.10, 0.10),
                    "long_term": np.random.uniform(-0.20, 0.20)
                },
                "momentum_indicators": {
                    "roc_20_day": np.random.uniform(-0.08, 0.08)
                },
                "data_quality": "mock_analysis"
            }
        
        context['task_instance'].xcom_push(key='sector_analysis', value=mock_analysis)
        
        return {
            "status": "success_mock",
            "sectors_analyzed": len(mock_analysis["sector_performance"]),
            "timestamp": datetime.now().isoformat()
        }

@task
def calculate_cross_asset_correlations(**context):
    """Calculate cross-asset correlations and detect regime changes."""
    try:
        import sys
        import os
        
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.core.correlation_engine import RealTimeCorrelationEngine
        
        # Get data from previous task
        market_data = context['task_instance'].xcom_pull(key='market_data')
        
        if not market_data:
            logger.error("No market data available for correlation analysis")
            return {"status": "error", "error": "No market data"}
        
        correlation_engine = RealTimeCorrelationEngine()
        
        # Prepare price data for correlation analysis
        price_data = {}
        
        for symbol, symbol_data in market_data.items():
            if "error" in symbol_data:
                continue
            
            # Use the most appropriate timeframe data
            if "daily" in symbol_data and isinstance(symbol_data["daily"], pd.DataFrame):
                price_data[symbol] = symbol_data["daily"]
            elif "hourly" in symbol_data and isinstance(symbol_data["hourly"], pd.DataFrame):
                price_data[symbol] = symbol_data["hourly"]
        
        if len(price_data) < 2:
            logger.error("Need at least 2 assets for correlation analysis")
            return {"status": "error", "error": "Insufficient assets for correlation"}
        
        # Calculate correlations
        correlation_analysis = correlation_engine.calculate_rolling_correlations(price_data)
        
        # Monitor cross-asset correlations
        cross_asset_analysis = correlation_engine.monitor_cross_asset_correlations(price_data)
        
        # Detect correlation breakdowns
        breakdown_analysis = correlation_engine.detect_correlation_breakdowns(price_data)
        
        # Combine all correlation analysis
        comprehensive_correlation = {
            "timestamp": datetime.now().isoformat(),
            "correlation_analysis": correlation_analysis,
            "cross_asset_analysis": cross_asset_analysis,
            "breakdown_analysis": breakdown_analysis,
            "summary": {
                "total_assets": len(price_data),
                "total_pairs": len(correlation_analysis.get("correlation_pairs", [])),
                "alerts_generated": len(correlation_analysis.get("alerts", [])),
                "breakdowns_detected": len(breakdown_analysis.get("breakdown_events", []))
            }
        }
        
        # Store results
        context['task_instance'].xcom_push(key='correlation_analysis', value=comprehensive_correlation)
        
        return {
            "status": "success",
            "assets_analyzed": len(price_data),
            "correlation_pairs": len(correlation_analysis.get("correlation_pairs", [])),
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError as e:
        logger.warning(f"Correlation analysis imports failed: {e}, using mock analysis")
        
        # Mock correlation analysis results
        market_data = context['task_instance'].xcom_pull(key='market_data') or {}
        symbols = list(market_data.keys())[:10]  # Limit to first 10 symbols
        
        mock_correlations = {}
        correlation_pairs = []
        
        # Generate mock correlation matrix
        for i, symbol1 in enumerate(symbols):
            mock_correlations[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    corr = 1.0
                else:
                    corr = np.random.uniform(-0.8, 0.9)
                mock_correlations[symbol1][symbol2] = corr
                
                if i < j:  # Avoid duplicates
                    correlation_pairs.append({
                        "pair": f"{symbol1}-{symbol2}",
                        "correlation": corr,
                        "regime": "moderate_positive" if corr > 0.3 else "low_correlation",
                        "cross_asset": True
                    })
        
        mock_comprehensive = {
            "timestamp": datetime.now().isoformat(),
            "correlation_analysis": {
                "correlation_matrix": mock_correlations,
                "correlation_pairs": correlation_pairs,
                "regime_analysis": {
                    "dominant_regime": "moderate_positive",
                    "market_stress_indicator": {
                        "stress_level": "low",
                        "stress_score": 0.2
                    }
                },
                "alerts": []
            },
            "cross_asset_analysis": {
                "diversification_analysis": {
                    "diversification_score": 75.0,
                    "portfolio_correlation": 0.45
                },
                "risk_on_risk_off": {
                    "current_regime": "neutral",
                    "regime_strength": 0.5
                }
            },
            "breakdown_analysis": {
                "breakdown_events": [],
                "stability_metrics": {
                    "stability_score": 0.8
                }
            },
            "summary": {
                "total_assets": len(symbols),
                "total_pairs": len(correlation_pairs),
                "alerts_generated": 0,
                "breakdowns_detected": 0
            }
        }
        
        context['task_instance'].xcom_push(key='correlation_analysis', value=mock_comprehensive)
        
        return {
            "status": "success_mock",
            "assets_analyzed": len(symbols),
            "correlation_pairs": len(correlation_pairs),
            "timestamp": datetime.now().isoformat()
        }

@task
def generate_rotation_signals(**context):
    """Generate sector rotation signals and portfolio recommendations."""
    try:
        import sys
        import os
        
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.core.sector_analysis import SectorRotationEngine
        
        # Get analysis results from previous tasks
        sector_analysis = context['task_instance'].xcom_pull(key='sector_analysis')
        correlation_analysis = context['task_instance'].xcom_pull(key='correlation_analysis')
        market_data = context['task_instance'].xcom_pull(key='market_data')
        
        if not sector_analysis or not market_data:
            logger.error("Missing required data for rotation signal generation")
            return {"status": "error", "error": "Missing analysis data"}
        
        rotation_engine = SectorRotationEngine()
        
        # Prepare price data
        price_data = {}
        for symbol, symbol_data in market_data.items():
            if "error" not in symbol_data:
                if "daily" in symbol_data and isinstance(symbol_data["daily"], pd.DataFrame):
                    price_data[symbol] = symbol_data["daily"]
        
        # Generate rotation strategy
        rotation_strategy = rotation_engine.generate_rotation_strategy(
            price_data, 
            current_allocation=None,  # No current allocation provided
            benchmark_data=price_data.get("SPY")
        )
        
        # Enhance with correlation insights
        if correlation_analysis and "correlation_analysis" in correlation_analysis:
            corr_data = correlation_analysis["correlation_analysis"]
            rotation_strategy["correlation_insights"] = {
                "regime_analysis": corr_data.get("regime_analysis", {}),
                "diversification_score": correlation_analysis.get("cross_asset_analysis", {}).get("diversification_analysis", {}).get("diversification_score", 50),
                "risk_regime": correlation_analysis.get("cross_asset_analysis", {}).get("risk_on_risk_off", {}).get("current_regime", "neutral")
            }
        
        # Store results
        context['task_instance'].xcom_push(key='rotation_signals', value=rotation_strategy)
        
        return {
            "status": "success",
            "rotation_recommendations": len(rotation_strategy.get("rotation_recommendations", {}).get("top_sectors_to_buy", [])),
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError as e:
        logger.warning(f"Rotation signal imports failed: {e}, using mock signals")
        
        # Mock rotation signals
        sector_analysis = context['task_instance'].xcom_pull(key='sector_analysis') or {}
        
        mock_rotation = {
            "timestamp": datetime.now().isoformat(),
            "rotation_recommendations": {
                "top_sectors_to_buy": [
                    {"symbol": "XLK", "sector": "Technology", "signal": "strong_buy", "rationale": "Strong momentum and relative performance"},
                    {"symbol": "XLV", "sector": "Healthcare", "signal": "buy", "rationale": "Defensive characteristics with growth potential"}
                ],
                "sectors_to_reduce": [
                    {"symbol": "XLE", "sector": "Energy", "signal": "sell", "rationale": "Weak momentum and high volatility"},
                    {"symbol": "XLU", "sector": "Utilities", "signal": "hold", "rationale": "Limited upside potential"}
                ],
                "overall_market_sentiment": "neutral"
            },
            "portfolio_optimization": {
                "recommended_allocation": {
                    "XLK": 0.20, "XLV": 0.15, "XLF": 0.12, "XLY": 0.10,
                    "SPY": 0.25, "TLT": 0.08, "GLD": 0.05, "VNQ": 0.05
                },
                "rebalancing_required": True,
                "diversification_score": 78.5
            },
            "risk_management": {
                "concentration_risk": {"max_single_allocation": 0.25, "concentration_level": "medium"},
                "correlation_risk": {"correlation_risk_level": "low"}
            },
            "correlation_insights": {
                "diversification_score": 75.0,
                "risk_regime": "neutral"
            }
        }
        
        context['task_instance'].xcom_push(key='rotation_signals', value=mock_rotation)
        
        return {
            "status": "success_mock",
            "rotation_recommendations": 2,
            "timestamp": datetime.now().isoformat()
        }

@task
def update_sector_correlation_database(**context):
    """Update database with sector and correlation analysis results."""
    try:
        import sys
        import os
        
        project_root = '/opt/airflow'
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.data.database import DatabaseManager
        
        # Get all analysis results
        sector_analysis = context['task_instance'].xcom_pull(key='sector_analysis') or {}
        correlation_analysis = context['task_instance'].xcom_pull(key='correlation_analysis') or {}
        rotation_signals = context['task_instance'].xcom_pull(key='rotation_signals') or {}
        
        db_manager = DatabaseManager()
        
        # Create database tables if they don't exist
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS sector_performance_history (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            sector VARCHAR(50),
            analysis_date DATE NOT NULL,
            short_term_return DECIMAL(8,6),
            medium_term_return DECIMAL(8,6),
            long_term_return DECIMAL(8,6),
            momentum_score DECIMAL(8,6),
            volatility DECIMAL(8,6),
            relative_strength DECIMAL(8,6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS correlation_matrix_daily (
            id SERIAL PRIMARY KEY,
            analysis_date DATE NOT NULL,
            asset1 VARCHAR(10) NOT NULL,
            asset2 VARCHAR(10) NOT NULL,
            correlation DECIMAL(8,6),
            correlation_regime VARCHAR(20),
            confidence DECIMAL(5,4),
            observations INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS rotation_signals_history (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            analysis_date DATE NOT NULL,
            rotation_signal VARCHAR(20),
            signal_strength DECIMAL(5,4),
            sector VARCHAR(50),
            recommended_allocation DECIMAL(5,4),
            rationale TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS market_regime_tracking (
            id SERIAL PRIMARY KEY,
            analysis_date DATE NOT NULL,
            correlation_regime VARCHAR(30),
            risk_regime VARCHAR(20),
            stress_level VARCHAR(10),
            diversification_score DECIMAL(5,2),
            market_sentiment VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_tables_sql)
            conn.commit()
        
        analysis_date = datetime.now().date()
        
        # Insert sector performance data
        sector_inserts = 0
        if "sector_performance" in sector_analysis:
            for symbol, performance in sector_analysis["sector_performance"].items():
                if "error" not in performance:
                    try:
                        insert_sql = """
                        INSERT INTO sector_performance_history 
                        (symbol, sector, analysis_date, short_term_return, medium_term_return, 
                         long_term_return, momentum_score, volatility, relative_strength)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        period_returns = performance.get("period_returns", {})
                        momentum = performance.get("momentum_indicators", {})
                        volatility_metrics = performance.get("volatility_metrics", {})
                        relative_perf = performance.get("relative_performance", {})
                        
                        with db_manager.get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(insert_sql, (
                                symbol,
                                performance.get("sector", "Unknown"),
                                analysis_date,
                                period_returns.get("short_term", 0),
                                period_returns.get("medium_term", 0),
                                period_returns.get("long_term", 0),
                                momentum.get("roc_20_day", 0),
                                volatility_metrics.get("volatility_20d", 0),
                                relative_perf.get("medium_term", 0)
                            ))
                            conn.commit()
                            sector_inserts += 1
                            
                    except Exception as e:
                        logger.error(f"Error inserting sector data for {symbol}: {e}")
        
        # Insert correlation data
        correlation_inserts = 0
        if ("correlation_analysis" in correlation_analysis and 
            "correlation_analysis" in correlation_analysis["correlation_analysis"]):
            
            corr_data = correlation_analysis["correlation_analysis"]
            correlation_pairs = corr_data.get("correlation_pairs", [])
            
            for pair in correlation_pairs:
                try:
                    insert_sql = """
                    INSERT INTO correlation_matrix_daily 
                    (analysis_date, asset1, asset2, correlation, correlation_regime, confidence, observations)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    assets = pair["pair"].split("-")
                    if len(assets) == 2:
                        with db_manager.get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(insert_sql, (
                                analysis_date,
                                assets[0],
                                assets[1],
                                pair.get("correlation", 0),
                                pair.get("regime", "unknown"),
                                pair.get("confidence", 0),
                                pair.get("observations", 0)
                            ))
                            conn.commit()
                            correlation_inserts += 1
                            
                except Exception as e:
                    logger.error(f"Error inserting correlation data for {pair.get('pair', 'unknown')}: {e}")
        
        # Insert rotation signals
        signal_inserts = 0
        if "rotation_recommendations" in rotation_signals:
            recommendations = rotation_signals["rotation_recommendations"]
            allocation = rotation_signals.get("portfolio_optimization", {}).get("recommended_allocation", {})
            
            # Process buy recommendations
            for rec in recommendations.get("top_sectors_to_buy", []):
                try:
                    insert_sql = """
                    INSERT INTO rotation_signals_history 
                    (symbol, analysis_date, rotation_signal, signal_strength, sector, 
                     recommended_allocation, rationale)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    with db_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(insert_sql, (
                            rec.get("symbol", ""),
                            analysis_date,
                            rec.get("signal", ""),
                            rec.get("strength", 0),
                            rec.get("sector", ""),
                            allocation.get(rec.get("symbol", ""), 0),
                            rec.get("rationale", "")
                        ))
                        conn.commit()
                        signal_inserts += 1
                        
                except Exception as e:
                    logger.error(f"Error inserting rotation signal for {rec.get('symbol', 'unknown')}: {e}")
        
        # Insert market regime data
        regime_inserts = 0
        try:
            insert_sql = """
            INSERT INTO market_regime_tracking 
            (analysis_date, correlation_regime, risk_regime, stress_level, 
             diversification_score, market_sentiment)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            # Extract regime information
            corr_regime = "unknown"
            risk_regime = "neutral"
            stress_level = "low"
            div_score = 50.0
            market_sentiment = "neutral"
            
            if ("correlation_analysis" in correlation_analysis and 
                "correlation_analysis" in correlation_analysis["correlation_analysis"]):
                regime_data = correlation_analysis["correlation_analysis"].get("regime_analysis", {})
                corr_regime = regime_data.get("dominant_regime", "unknown")
                stress_data = regime_data.get("market_stress_indicator", {})
                stress_level = stress_data.get("stress_level", "low")
            
            if ("cross_asset_analysis" in correlation_analysis and 
                "cross_asset_analysis" in correlation_analysis):
                cross_data = correlation_analysis["cross_asset_analysis"]
                risk_regime = cross_data.get("risk_on_risk_off", {}).get("current_regime", "neutral")
                div_score = cross_data.get("diversification_analysis", {}).get("diversification_score", 50.0)
            
            if "rotation_recommendations" in rotation_signals:
                market_sentiment = rotation_signals["rotation_recommendations"].get("overall_market_sentiment", "neutral")
            
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(insert_sql, (
                    analysis_date,
                    corr_regime,
                    risk_regime,
                    stress_level,
                    div_score,
                    market_sentiment
                ))
                conn.commit()
                regime_inserts = 1
                
        except Exception as e:
            logger.error(f"Error inserting market regime data: {e}")
        
        return {
            "status": "success",
            "sector_inserts": sector_inserts,
            "correlation_inserts": correlation_inserts,
            "signal_inserts": signal_inserts,
            "regime_inserts": regime_inserts,
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError as e:
        logger.warning(f"Database imports failed: {e}, skipping database update")
        return {"status": "skipped", "reason": "Database imports failed"}
    except Exception as e:
        logger.error(f"Error updating sector correlation database: {e}")
        return {"status": "error", "error": str(e)}

# Task creation and dependencies - wrap in try/catch for Airflow compatibility
try:
    with dag:
        # Market session check (branching)
        session_check = BranchPythonOperator(
            task_id="check_market_session",
            python_callable=check_market_session,
            dag=dag,
        )
        
        # Data collection tasks (branched based on market session)
        comprehensive_task = collect_comprehensive_data()
        intraday_task = collect_intraday_data()
        premarket_task = collect_premarket_data()
        
        # Join point after branching
        join_task = DummyOperator(
            task_id="join_data_collection",
            trigger_rule="none_failed_or_skipped",
            dag=dag,
        )
        
        # Analysis tasks (run in parallel after data collection)
        sector_analysis_task = analyze_sector_performance()
        correlation_analysis_task = calculate_cross_asset_correlations()
        
        # Signal generation (depends on both analyses)
        rotation_signals_task = generate_rotation_signals()
        
        # Database update
        db_update_task = update_sector_correlation_database()
        
        # Set up dependencies
        session_check >> [comprehensive_task, intraday_task, premarket_task]
        [comprehensive_task, intraday_task, premarket_task] >> join_task
        join_task >> [sector_analysis_task, correlation_analysis_task]
        [sector_analysis_task, correlation_analysis_task] >> rotation_signals_task
        rotation_signals_task >> db_update_task

except Exception as e:
    # During testing or import issues, skip dependency setup
    logger.warning(f"Failed to set up sector correlation DAG dependencies: {e}")
    pass