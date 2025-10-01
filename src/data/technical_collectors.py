"""Technical indicator data collection and caching system."""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import json
from functools import wraps

from src.config import settings
from src.core.technical_analysis import MultiTimeframeAnalysis
from src.data.collectors import MarketDataCollector

# Try to import redis for caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis not available, caching will use memory only")

# Try to import yfinance for historical data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available, using dummy data")

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for technical indicator data."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.redis_client = None
        self.memory_cache = {}
        self.cache_expiry = {
            "1m": 60,      # 1 minute data expires in 1 minute
            "5m": 300,     # 5 minute data expires in 5 minutes
            "15m": 900,    # 15 minute data expires in 15 minutes
            "1h": 3600,    # 1 hour data expires in 1 hour
            "4h": 14400,   # 4 hour data expires in 4 hours
            "1d": 86400    # 1 day data expires in 1 day
        }
        
        # Initialize Redis if available
        if REDIS_AVAILABLE and settings.use_real_data:
            try:
                self.redis_client = redis.Redis.from_url(settings.redis_url)
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using memory cache")
                self.redis_client = None
    
    def get_cache_key(self, symbol: str, timeframe: str, indicator_type: str = "indicators") -> str:
        """Generate cache key for indicator data.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            indicator_type: Type of data being cached
            
        Returns:
            Cache key string
        """
        return f"technical:{symbol}:{timeframe}:{indicator_type}"
    
    def get(self, key: str) -> Optional[Dict]:
        """Get data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data.decode('utf-8'))
            else:
                # Memory cache
                if key in self.memory_cache:
                    cached_item = self.memory_cache[key]
                    if datetime.now() < cached_item["expires_at"]:
                        return cached_item["data"]
                    else:
                        # Remove expired item
                        del self.memory_cache[key]
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
        
        return None
    
    def set(self, key: str, data: Dict, timeframe: str) -> bool:
        """Set data in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            timeframe: Timeframe for expiry calculation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            ttl = self.cache_expiry.get(timeframe, 3600)
            
            # Add metadata
            cache_data = {
                "data": data,
                "cached_at": datetime.now().isoformat(),
                "timeframe": timeframe
            }
            
            if self.redis_client:
                self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(cache_data, default=str)
                )
            else:
                # Memory cache
                self.memory_cache[key] = {
                    "data": cache_data,
                    "expires_at": datetime.now() + timedelta(seconds=ttl)
                }
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match (e.g., 'technical:AAPL:*')
            
        Returns:
            Number of keys invalidated
        """
        count = 0
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    count = self.redis_client.delete(*keys)
            else:
                # Memory cache
                keys_to_delete = [k for k in self.memory_cache.keys() if self._match_pattern(k, pattern)]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                count = len(keys_to_delete)
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
        
        return count
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for memory cache."""
        pattern = pattern.replace('*', '.*')
        import re
        return bool(re.match(pattern, key))
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "cache_type": "redis" if self.redis_client else "memory",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if self.redis_client:
                info = self.redis_client.info()
                stats.update({
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                })
            else:
                stats.update({
                    "memory_cache_entries": len(self.memory_cache),
                    "memory_cache_keys": list(self.memory_cache.keys())[:10]  # First 10 keys
                })
        except Exception as e:
            stats["error"] = str(e)
        
        return stats


def cache_result(timeframe: str):
    """Decorator to cache technical indicator results.
    
    Args:
        timeframe: Timeframe for cache expiry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, symbol: str, *args, **kwargs):
            if not hasattr(self, 'cache_manager'):
                return func(self, symbol, *args, **kwargs)
            
            # Generate cache key
            cache_key = self.cache_manager.get_cache_key(
                symbol, timeframe, func.__name__
            )
            
            # Try to get from cache first
            cached_result = self.cache_manager.get(cache_key)
            if cached_result and cached_result.get("data"):
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result["data"]
            
            # Calculate and cache result
            result = func(self, symbol, *args, **kwargs)
            if result:
                self.cache_manager.set(cache_key, result, timeframe)
                logger.debug(f"Cached result for {cache_key}")
            
            return result
        return wrapper
    return decorator


class TechnicalDataCollector:
    """Collects and caches technical indicator data across multiple timeframes."""
    
    def __init__(self):
        """Initialize technical data collector."""
        self.cache_manager = CacheManager()
        self.market_data_collector = MarketDataCollector()
        self.analysis_engine = MultiTimeframeAnalysis()
        
        # Data collection settings
        self.max_historical_days = 100
        self.min_data_points = {
            "1m": 50,
            "5m": 50,
            "15m": 50,
            "1h": 50,
            "4h": 30,
            "1d": 20
        }
        
        # Retry settings
        self.retry_attempts = 3
        self.retry_delay = 1
    
    def collect_technical_indicators(self, symbol: str, 
                                   timeframes: List[str] = None) -> Dict[str, Dict]:
        """Collect technical indicators for a symbol across multiple timeframes.
        
        Args:
            symbol: Stock symbol
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary with timeframe indicators and metadata
        """
        if timeframes is None:
            timeframes = ['5m', '1h', '1d']
        
        logger.info(f"Collecting technical indicators for {symbol} on timeframes: {timeframes}")
        
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "timeframes": {},
            "cache_info": {},
            "data_quality": {}
        }
        
        for timeframe in timeframes:
            try:
                logger.debug(f"Processing {symbol} for {timeframe} timeframe")
                
                # Get historical data for the timeframe
                historical_data = self._get_historical_data(symbol, timeframe)
                
                if historical_data is None or historical_data.empty:
                    logger.warning(f"No historical data available for {symbol} {timeframe}")
                    results["timeframes"][timeframe] = {"error": "No historical data"}
                    continue
                
                # Calculate indicators
                indicators = self.analysis_engine.calculate_indicators(
                    historical_data, [timeframe]
                )
                
                if timeframe in indicators:
                    results["timeframes"][timeframe] = indicators[timeframe]
                    
                    # Add data quality metrics
                    results["data_quality"][timeframe] = self._assess_data_quality(
                        historical_data, timeframe
                    )
                else:
                    results["timeframes"][timeframe] = {"error": "Indicator calculation failed"}
                
            except Exception as e:
                logger.error(f"Error collecting indicators for {symbol} {timeframe}: {e}")
                results["timeframes"][timeframe] = {"error": str(e)}
        
        # Add cache statistics
        results["cache_info"] = self.cache_manager.get_cache_stats()
        
        return results
    
    def _get_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data for indicator calculation.
        
        Args:
            symbol: Stock symbol
            timeframe: Target timeframe
            
        Returns:
            Historical OHLCV data as DataFrame
        """
        cache_key = self.cache_manager.get_cache_key(symbol, timeframe, "historical")
        
        # Check cache first
        cached_data = self.cache_manager.get(cache_key)
        if cached_data and cached_data.get("data"):
            logger.debug(f"Using cached historical data for {symbol} {timeframe}")
            try:
                # Convert back to DataFrame
                df_data = cached_data["data"]
                df = pd.DataFrame(df_data["data"])
                df.index = pd.to_datetime(df_data["index"])
                return df
            except Exception as e:
                logger.warning(f"Error loading cached data: {e}")
        
        # Fetch fresh data
        if not settings.use_real_data or not YFINANCE_AVAILABLE:
            historical_data = self._generate_dummy_historical_data(symbol, timeframe)
        else:
            historical_data = self._fetch_yfinance_historical_data(symbol, timeframe)
        
        # Cache the data
        if historical_data is not None and not historical_data.empty:
            try:
                # Convert DataFrame to serializable format
                cache_data = {
                    "data": historical_data.to_dict('records'),
                    "index": historical_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    "columns": historical_data.columns.tolist()
                }
                self.cache_manager.set(cache_key, cache_data, timeframe)
            except Exception as e:
                logger.warning(f"Error caching historical data: {e}")
        
        return historical_data
    
    def _fetch_yfinance_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch historical data using yfinance.
        
        Args:
            symbol: Stock symbol
            timeframe: Target timeframe
            
        Returns:
            OHLCV data as DataFrame
        """
        for attempt in range(self.retry_attempts):
            try:
                ticker = yf.Ticker(symbol)
                
                # Map timeframe to yfinance intervals and periods
                interval_map = {
                    "1m": {"interval": "1m", "period": "1d"},      # 1 day of 1-minute data
                    "5m": {"interval": "5m", "period": "5d"},      # 5 days of 5-minute data
                    "15m": {"interval": "15m", "period": "1mo"},   # 1 month of 15-minute data
                    "1h": {"interval": "1h", "period": "3mo"},     # 3 months of hourly data
                    "4h": {"interval": "1h", "period": "1y"},      # 1 year hourly, resample to 4h
                    "1d": {"interval": "1d", "period": "2y"}       # 2 years of daily data
                }
                
                config = interval_map.get(timeframe, interval_map["1d"])
                
                # Fetch data
                hist = ticker.history(
                    period=config["period"],
                    interval=config["interval"]
                )
                
                if hist.empty:
                    logger.warning(f"No data returned from yfinance for {symbol} {timeframe}")
                    if attempt == self.retry_attempts - 1:
                        return self._generate_dummy_historical_data(symbol, timeframe)
                    continue
                
                # Resample 1h to 4h if needed
                if timeframe == "4h" and config["interval"] == "1h":
                    hist = hist.resample('4H').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                
                # Ensure we have enough data points
                min_points = self.min_data_points.get(timeframe, 50)
                if len(hist) < min_points:
                    logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(hist)} < {min_points}")
                    if attempt == self.retry_attempts - 1:
                        return self._generate_dummy_historical_data(symbol, timeframe)
                    continue
                
                # Clean and validate data
                hist = hist.dropna()
                hist = hist[hist['Volume'] > 0]  # Remove zero volume periods
                
                return hist
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol} {timeframe}: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to fetch data for {symbol} {timeframe}")
                    return self._generate_dummy_historical_data(symbol, timeframe)
        
        return None
    
    def _generate_dummy_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate dummy historical OHLCV data for testing.
        
        Args:
            symbol: Stock symbol
            timeframe: Target timeframe
            
        Returns:
            Dummy OHLCV data as DataFrame
        """
        import random
        
        # Base prices for different symbols
        base_prices = {
            "SPY": 450.0, "QQQ": 380.0, "AAPL": 180.0, "MSFT": 340.0,
            "TSLA": 240.0, "GOOGL": 140.0, "AMZN": 150.0, "META": 320.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Determine number of periods and time delta
        periods_map = {
            "1m": {"count": 100, "freq": "1min"},
            "5m": {"count": 100, "freq": "5min"},
            "15m": {"count": 100, "freq": "15min"},
            "1h": {"count": 100, "freq": "1h"},
            "4h": {"count": 60, "freq": "4h"},
            "1d": {"count": 100, "freq": "1D"}
        }
        
        config = periods_map.get(timeframe, periods_map["1d"])
        
        # Generate time index
        end_time = datetime.now()
        periods = pd.date_range(
            end=end_time,
            periods=config["count"],
            freq=config["freq"]
        )
        
        # Generate price data with realistic movements
        prices = []
        current_price = base_price
        
        for i in range(len(periods)):
            # Random walk with mean reversion
            change_pct = random.gauss(0, 0.01)  # 1% std dev
            current_price *= (1 + change_pct)
            
            # Add some mean reversion
            if current_price > base_price * 1.1:
                current_price *= 0.999
            elif current_price < base_price * 0.9:
                current_price *= 1.001
            
            prices.append(current_price)
        
        # Generate OHLCV data
        data = []
        for i, price in enumerate(prices):
            daily_range = price * random.uniform(0.01, 0.03)  # 1-3% daily range
            
            high = price + random.uniform(0, daily_range / 2)
            low = price - random.uniform(0, daily_range / 2)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            # Ensure OHLC logic
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            volume = random.randint(100000, 5000000)
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=periods)
        return df
    
    def _assess_data_quality(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """Assess quality of historical data.
        
        Args:
            data: Historical OHLCV data
            timeframe: Timeframe being assessed
            
        Returns:
            Data quality metrics
        """
        if data.empty:
            return {"quality": "poor", "issues": ["No data available"]}
        
        quality_metrics = {
            "total_periods": len(data),
            "missing_periods": data.isnull().sum().sum(),
            "zero_volume_periods": (data['Volume'] == 0).sum(),
            "data_completeness": 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            "timeframe": timeframe,
            "data_range": {
                "start": data.index.min().isoformat(),
                "end": data.index.max().isoformat()
            }
        }
        
        # Assess overall quality
        issues = []
        if quality_metrics["total_periods"] < self.min_data_points.get(timeframe, 50):
            issues.append("Insufficient data points")
        
        if quality_metrics["data_completeness"] < 0.95:
            issues.append("High missing data percentage")
        
        if quality_metrics["zero_volume_periods"] > len(data) * 0.1:
            issues.append("Many zero volume periods")
        
        # Determine quality rating
        if not issues:
            quality = "excellent"
        elif len(issues) == 1:
            quality = "good"
        elif len(issues) == 2:
            quality = "fair"
        else:
            quality = "poor"
        
        quality_metrics["quality"] = quality
        quality_metrics["issues"] = issues
        
        return quality_metrics
    
    @cache_result("5m")
    def get_realtime_indicators(self, symbol: str) -> Dict:
        """Get real-time technical indicators with 5-minute caching.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Real-time indicators
        """
        return self.collect_technical_indicators(symbol, ["1m", "5m"])
    
    @cache_result("1h") 
    def get_hourly_indicators(self, symbol: str) -> Dict:
        """Get hourly technical indicators with 1-hour caching.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Hourly indicators
        """
        return self.collect_technical_indicators(symbol, ["15m", "1h"])
    
    @cache_result("1d")
    def get_daily_indicators(self, symbol: str) -> Dict:
        """Get daily technical indicators with 1-day caching.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Daily indicators
        """
        return self.collect_technical_indicators(symbol, ["4h", "1d"])
    
    def invalidate_symbol_cache(self, symbol: str) -> int:
        """Invalidate all cached data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Number of cache entries invalidated
        """
        pattern = f"technical:{symbol}:*"
        count = self.cache_manager.invalidate(pattern)
        logger.info(f"Invalidated {count} cache entries for {symbol}")
        return count
    
    def get_cache_status(self) -> Dict:
        """Get current cache status and statistics.
        
        Returns:
            Cache status information
        """
        return self.cache_manager.get_cache_stats()
    
    def bulk_collect_indicators(self, symbols: List[str], 
                              timeframes: List[str] = None) -> Dict[str, Dict]:
        """Collect indicators for multiple symbols efficiently.
        
        Args:
            symbols: List of stock symbols
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary with symbol as key and indicators as value
        """
        if timeframes is None:
            timeframes = ['5m', '1h', '1d']
        
        logger.info(f"Bulk collecting indicators for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.collect_technical_indicators(symbol, timeframes)
            except Exception as e:
                logger.error(f"Error collecting indicators for {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        return results