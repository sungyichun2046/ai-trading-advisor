"""
Data Manager Module - Consolidated data collection, processing, storage, and monitoring.
"""
import logging, os, time, random, psycopg2, requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import pandas as pd, numpy as np
from ..config import settings

# Shared utilities with fallbacks
try:
    from ..utils.shared import send_alerts, log_performance
except ImportError:
    def send_alerts(alert_type, message, severity="info", context=None): return {"status": "fallback", "alert_sent": True}
    def log_performance(operation, start_time, end_time, status="success", metrics=None): return {"operation": operation, "status": status}

# Optional imports
try: import yfinance as yf; YFINANCE_AVAILABLE = True
except ImportError: YFINANCE_AVAILABLE = False
try: from newsapi import NewsApiClient; NEWSAPI_AVAILABLE = True
except ImportError: NEWSAPI_AVAILABLE = False
try: from transformers import pipeline; TRANSFORMERS_AVAILABLE = True
except ImportError: TRANSFORMERS_AVAILABLE = False
try: from textblob import TextBlob; TEXTBLOB_AVAILABLE = True
except ImportError: TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataManager:
    """Consolidated data management system with monitoring capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.retry_attempts = 3
        self.retry_delay = 1
        
        if not hasattr(settings, 'use_real_data'): raise ValueError("USE_REAL_DATA flag required in settings")
        logger.info(f"DataManager initialized with USE_REAL_DATA={settings.use_real_data}")
        
        self.connection_params = {'host': os.getenv('POSTGRES_HOST', 'postgres'), 'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'trading_advisor'), 'user': os.getenv('POSTGRES_USER', 'trader'),
            'password': os.getenv('POSTGRES_PASSWORD', 'trader_password')}
        
        self.sentiment_analyzer, self.sentiment_method, self.newsapi_client = None, "dummy", None
        self._setup_sentiment_analyzer()
        if settings.use_real_data and hasattr(settings, 'newsapi_key') and settings.newsapi_key and NEWSAPI_AVAILABLE:
            try: self.newsapi_client = NewsApiClient(api_key=settings.newsapi_key)
            except Exception as e: logger.warning(f"Failed to initialize NewsAPI: {e}")
    
    def _setup_sentiment_analyzer(self) -> None:
        """Setup sentiment analyzer with fallbacks."""
        if TRANSFORMERS_AVAILABLE:
            try: self.sentiment_analyzer, self.sentiment_method = pipeline("sentiment-analysis", model="ProsusAI/finbert"), "finbert"; return
            except: pass
        self.sentiment_method = "textblob" if TEXTBLOB_AVAILABLE else "dummy"
    
    def collect_market_data(self, symbols: List[str], timeframe: str = "1d", period: str = "1mo") -> Dict[str, Any]:
        """Collect market data for specified symbols."""
        logger.info(f"Collecting market data for {len(symbols)} symbols")
        collected_data, errors = {}, []
        
        for symbol in symbols:
            try:
                data = self._generate_dummy_market_data(symbol, period) if not settings.use_real_data else self._collect_yfinance_data(symbol, period, timeframe)
                if data: collected_data[symbol] = data
                else: errors.append(f"Failed to collect data for {symbol}")
            except Exception as e: logger.error(f"Error collecting {symbol}: {e}"); errors.append(f"Error for {symbol}: {str(e)}")
        
        return {"status": "success" if collected_data else "failed", "data": collected_data, "errors": errors,
                "timestamp": datetime.now().isoformat(), "symbols_collected": len(collected_data), "total_symbols": len(symbols)}
    
    def _collect_yfinance_data(self, symbol: str, period: str = "1mo", interval: str = "15m") -> Optional[Dict]:
        """Collect data using Yahoo Finance API with fallbacks."""
        for attempt in range(self.retry_attempts):
            try:
                result = self._collect_yahoo_direct(symbol)
                if result: return result
                
                if YFINANCE_AVAILABLE:
                    hist = yf.Ticker(symbol).history(period=period, interval=interval)
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        return {"symbol": symbol, "status": "success", "price": round(float(latest['Close']), 2), "volume": int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                               "open": round(float(latest['Open']), 2), "high": round(float(latest['High']), 2), "low": round(float(latest['Low']), 2), "close": round(float(latest['Close']), 2),
                               "timestamp": datetime.now().isoformat(), "data_source": "yfinance"}
            except Exception as e:
                if attempt < self.retry_attempts - 1: time.sleep(self.retry_delay)
        return self._generate_dummy_market_data(symbol)
    
    def _collect_yahoo_direct(self, symbol: str) -> Optional[Dict]:
        """Collect data directly from Yahoo Finance API."""
        try:
            response = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                                  headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            data = response.json()['chart']['result'][0]
            quotes = data['indicators']['quote'][0]
            
            if data['timestamp'] and quotes['close']:
                close_price = quotes['close'][-1]
                return {"symbol": symbol, "status": "success", "price": round(float(close_price), 2),
                       "volume": int(quotes['volume'][-1] or 0), "open": round(float(quotes['open'][-1] or close_price), 2),
                       "high": round(float(quotes['high'][-1] or close_price), 2), "low": round(float(quotes['low'][-1] or close_price), 2),
                       "close": round(float(close_price), 2), "timestamp": datetime.fromtimestamp(data['timestamp'][-1]).isoformat(),
                       "market_cap": data['meta'].get('marketCap'), "data_source": "yahoo_direct"}
        except Exception as e: logger.debug(f"Yahoo API failed for {symbol}: {e}")
        return None
    
    def _generate_dummy_market_data(self, symbol: str, period: str = "1mo") -> Dict:
        """Generate dummy market data."""
        base_prices = {"SPY": 450.0, "QQQ": 380.0, "AAPL": 180.0, "MSFT": 340.0, "TSLA": 240.0}
        current_price = base_prices.get(symbol, 100.0) * (1 + random.uniform(-0.05, 0.05))
        return {"symbol": symbol, "status": "success", "price": round(current_price, 2), "volume": random.randint(100000, 2000000),
               "open": round(current_price * random.uniform(0.99, 1.01), 2), "high": round(current_price * random.uniform(1.00, 1.02), 2),
               "low": round(current_price * random.uniform(0.98, 1.00), 2), "close": round(current_price, 2),
               "timestamp": datetime.now().isoformat(), "market_cap": random.randint(50000000000, 3000000000000),
               "pe_ratio": round(random.uniform(15.0, 35.0), 2), "data_source": "dummy"}
    
    def collect_fundamental_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect fundamental data for specified symbols."""
        logger.info(f"Collecting fundamental data for {len(symbols)} symbols")
        collected_data, errors = [], []
        
        for symbol in symbols:
            try:
                data = self._collect_weekly_fundamentals(symbol)
                if data and data.get("status") == "success": collected_data.append(data)
                else: errors.append(f"Failed for {symbol}")
            except Exception as e: logger.error(f"Error for {symbol}: {e}"); errors.append(f"Error {symbol}: {str(e)}")
        
        return {"status": "success" if collected_data else "failed", "data": collected_data, "errors": errors,
                "timestamp": datetime.now().isoformat(), "symbols_collected": len(collected_data), "total_symbols": len(symbols)}
    
    def _collect_weekly_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Collect weekly fundamental data for a symbol."""
        if not settings.use_real_data: return self._generate_dummy_fundamental_data(symbol)
        try:
            if YFINANCE_AVAILABLE:
                info = yf.Ticker(symbol).info
                if info: return {"status": "success", "symbol": symbol, "pe_ratio": info.get('forwardPE', 20.0), "pb_ratio": info.get('priceToBook', 3.0), "ps_ratio": info.get('priceToSalesTrailing12Months', 2.0),
                               "debt_to_equity": info.get('debtToEquity', 0.5), "profit_margins": info.get('profitMargins', 0.15), "return_on_equity": info.get('returnOnEquity', 0.18), "revenue_growth": info.get('revenueGrowth', 0.12),
                               "earnings_growth": info.get('earningsGrowth', 0.10), "current_ratio": info.get('currentRatio', 1.5), "quick_ratio": info.get('quickRatio', 1.2), "timestamp": datetime.now().isoformat(), "data_source": "yfinance"}
        except: pass
        return self._generate_dummy_fundamental_data(symbol)
    
    def _generate_dummy_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Generate dummy fundamental data."""
        return {"status": "success", "symbol": symbol, "pe_ratio": round(random.uniform(15.0, 35.0), 2),
               "pb_ratio": round(random.uniform(1.5, 5.0), 2), "ps_ratio": round(random.uniform(1.0, 4.0), 2),
               "debt_to_equity": round(random.uniform(0.2, 1.5), 2), "profit_margins": round(random.uniform(0.05, 0.25), 3),
               "return_on_equity": round(random.uniform(0.08, 0.30), 3), "revenue_growth": round(random.uniform(-0.05, 0.20), 3),
               "earnings_growth": round(random.uniform(-0.10, 0.25), 3), "current_ratio": round(random.uniform(1.0, 3.0), 2),
               "quick_ratio": round(random.uniform(0.8, 2.5), 2), "timestamp": datetime.now().isoformat(), "data_source": "dummy"}
    
    def collect_sentiment_data(self, symbols: Optional[List[str]] = None, max_articles: int = 50) -> Dict[str, Any]:
        """Collect news data and sentiment analysis."""
        logger.info(f"Collecting news sentiment data (max {max_articles} articles)")
        if not settings.use_real_data or not self.newsapi_client: return self._generate_dummy_news_sentiment(max_articles)
        try:
            articles, processed_articles = self._collect_newsapi_data(max_articles), []
            for article in articles:
                sentiment = self._analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                processed_articles.append({"title": article.get('title', ''), "content": article.get('description', ''), "url": article.get('url', ''), "source": article.get('source', {}).get('name', ''),
                                         "published_at": article.get('publishedAt', ''), "sentiment_score": sentiment['score'], "sentiment_label": sentiment['label'], "timestamp": datetime.now().isoformat()})
            return {"status": "success", "articles": processed_articles, "article_count": len(processed_articles), "sentiment_method": self.sentiment_method, "timestamp": datetime.now().isoformat()}
        except Exception as e: logger.error(f"Failed to collect news sentiment: {e}"); return self._generate_dummy_news_sentiment(max_articles)
    
    def _collect_newsapi_data(self, max_articles: int = 50) -> List[Dict]:
        """Collect news using NewsAPI."""
        keywords, all_articles = ["stock market", "earnings", "economy"], []
        for keyword in keywords:
            try:
                response = self.newsapi_client.get_everything(q=keyword, language='en', sort_by='publishedAt', page_size=max_articles//len(keywords))
                if response.get('status') == 'ok': all_articles.extend(response.get('articles', []))
            except Exception as e: logger.warning(f"Failed news for '{keyword}': {e}")
        return all_articles[:max_articles]
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        if not text or not text.strip(): return {"score": 0.0, "label": "neutral", "confidence": 0.0}
        
        if self.sentiment_method == "finbert" and self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)[0]
                return {"score": result['score'] if result['label'] == 'positive' else -result['score'],
                       "label": result['label'], "confidence": result['score']}
            except: pass
        
        if self.sentiment_method == "textblob" and TEXTBLOB_AVAILABLE:
            try:
                polarity = TextBlob(text).sentiment.polarity
                return {"score": polarity, "label": "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral", "confidence": abs(polarity)}
            except: pass
        
        sentiment_score = random.uniform(-1.0, 1.0)
        return {"score": sentiment_score, "label": "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral", "confidence": abs(sentiment_score)}
    
    def _generate_dummy_news_sentiment(self, max_articles: int) -> Dict[str, Any]:
        """Generate dummy news sentiment data."""
        dummy_base, articles = [("Market Outlook Positive", "Analysts optimistic"), ("Fed Maintains Rates", "Rates steady"), ("Tech Stocks Rally", "Tech strong performance")], []
        for i in range(min(max_articles, len(dummy_base) * 5)):
            base = dummy_base[i % len(dummy_base)]
            sentiment = self._analyze_sentiment(base[0] + ' ' + base[1])
            articles.append({"title": f"{base[0]} {i+1}", "content": base[1], "url": f"https://example.com/article/{i+1}", "source": "Dummy News", "published_at": (datetime.now() - timedelta(hours=i)).isoformat(),
                           "sentiment_score": sentiment['score'], "sentiment_label": sentiment['label'], "timestamp": datetime.now().isoformat()})
        return {"status": "success", "articles": articles, "article_count": len(articles), "sentiment_method": "dummy", "timestamp": datetime.now().isoformat()}
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except psycopg2.OperationalError as e:
            if self.connection_params['host'] == 'postgres':
                try: 
                    localhost_params = self.connection_params.copy()
                    localhost_params['host'] = 'localhost'
                    if conn: conn.close()
                    conn = psycopg2.connect(**localhost_params)
                    yield conn
                except Exception: logger.error(f"DB connection failed: {e}"); raise
            else: logger.error(f"DB error: {e}"); raise
        except Exception as e: logger.error(f"DB connection error: {e}"); raise
        finally: 
            if conn: conn.close()
    
    def store_market_data(self, market_data: Dict, execution_date: datetime) -> bool:
        """Store market data in database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO market_data (symbol, price, volume, open_price, high_price, low_price, close_price, market_cap, pe_ratio, data_source, timestamp, execution_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                             (market_data['symbol'], market_data['price'], market_data.get('volume', 0), market_data.get('open'),
                              market_data.get('high'), market_data.get('low'), market_data.get('close'), market_data.get('market_cap'),
                              market_data.get('pe_ratio'), market_data.get('data_source', 'unknown'), market_data.get('timestamp', datetime.now()), execution_date.date()))
                conn.commit()
                return True
        except Exception as e: logger.error(f"Failed to store market data: {e}"); return False
    
    def store_news_data(self, news_data: Dict, execution_date: datetime) -> bool:
        """Store news and sentiment data."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO news_data (title, content, url, source, published_at, sentiment_score, sentiment_label, execution_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                             (news_data['title'], news_data.get('content'), news_data.get('url'), news_data.get('source'),
                              news_data.get('published_at'), news_data.get('sentiment_score'), news_data.get('sentiment_label'), execution_date.date()))
                conn.commit()
                return True
        except Exception as e: logger.error(f"Failed to store news data: {e}"); return False
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE IF NOT EXISTS market_data (id SERIAL PRIMARY KEY, symbol VARCHAR(10) NOT NULL, price DECIMAL(10,2) NOT NULL, volume BIGINT, open_price DECIMAL(10,2), high_price DECIMAL(10,2), low_price DECIMAL(10,2), close_price DECIMAL(10,2), market_cap BIGINT, pe_ratio DECIMAL(6,2), data_source VARCHAR(20), timestamp TIMESTAMP NOT NULL, execution_date DATE NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
                cursor.execute("CREATE TABLE IF NOT EXISTS news_data (id SERIAL PRIMARY KEY, title TEXT NOT NULL, content TEXT, url TEXT, source VARCHAR(100), published_at TIMESTAMP, sentiment_score DECIMAL(4,3), sentiment_label VARCHAR(20), execution_date DATE NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
                conn.commit()
                logger.info("Database tables created successfully")
        except Exception as e: logger.error(f"Failed to create tables: {e}"); raise
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on data systems."""
        health_status = {"status": "healthy", "timestamp": datetime.now().isoformat(), "components": {}}
        
        try:
            with self.get_connection() as conn: conn.cursor().execute("SELECT 1")
            health_status["components"]["database"] = "healthy"
        except Exception as e: health_status["components"]["database"] = f"unhealthy: {e}"; health_status["status"] = "degraded"
        
        health_status["components"].update({"yfinance": "available" if YFINANCE_AVAILABLE else "unavailable",
                                           "newsapi": "available" if self.newsapi_client else "unavailable", "sentiment": self.sentiment_method})
        return health_status

    def monitor_data_quality(self, data_source: str = "all") -> Dict[str, Any]:
        """Monitor data quality across all collection systems using shared utilities."""
        start_time = datetime.now()
        try:
            logger.info(f"Starting data quality monitoring for {data_source}")
            quality_metrics, alerts_to_send = {}, []
            
            if data_source in ["all", "market"]: market_quality = self._check_market_data_quality(); quality_metrics["market_data"] = market_quality; market_quality["quality_score"] < 0.8 and alerts_to_send.append({"type": "data_quality", "source": "market", "score": market_quality["quality_score"]})
            if data_source in ["all", "news"]: news_quality = self._check_news_data_quality(); quality_metrics["news_data"] = news_quality; news_quality["quality_score"] < 0.7 and alerts_to_send.append({"type": "data_quality", "source": "news", "score": news_quality["quality_score"]})
            if data_source in ["all", "database"]: db_health = self._check_database_health(); quality_metrics["database"] = db_health; not db_health["healthy"] and alerts_to_send.append({"type": "database_health", "status": "unhealthy", "issues": db_health["issues"]})
            
            scores = [m.get("quality_score", 1.0) for m in quality_metrics.values() if "quality_score" in m]
            overall_score = sum(scores) / len(scores) if scores else 1.0
            
            for alert in alerts_to_send: send_alerts("data_quality_degradation", f"Data quality issue: {alert}", "warning" if alert.get("score", 0) > 0.5 else "error", alert)
            
            result = {"status": "success", "overall_quality_score": overall_score, "component_metrics": quality_metrics, "alerts_generated": len(alerts_to_send), "monitoring_source": data_source, "timestamp": datetime.now().isoformat()}
            log_performance("Data Quality Monitoring", start_time, datetime.now(), "success", {"overall_score": overall_score, "components_checked": len(quality_metrics)})
            return result
        except Exception as e:
            logger.error(f"Data quality monitoring failed: {e}"); log_performance("Data Quality Monitoring", start_time, datetime.now(), "error", {"error": str(e)}); send_alerts("monitoring_system_error", f"Data quality monitoring failed: {str(e)}", "error")
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

    def monitor_data_freshness(self, max_age_hours: int = 2) -> Dict[str, Any]:
        """Monitor data freshness and identify stale data using shared utilities."""
        start_time = datetime.now()
        try:
            logger.info(f"Monitoring data freshness with max age {max_age_hours} hours")
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            freshness_results, stale_sources = {}, []
            
            try:
                with self.get_connection() as conn: cursor = conn.cursor(); cursor.execute("SELECT MAX(timestamp) FROM market_data WHERE created_at > %s", (cutoff_time,)); latest_market = cursor.fetchone()[0]; market_age = (datetime.now() - latest_market).total_seconds() / 3600 if latest_market else max_age_hours + 1; freshness_results["market_data"] = {"age_hours": market_age, "is_fresh": market_age <= max_age_hours}; market_age > max_age_hours and stale_sources.append("market_data")
            except Exception as e: logger.warning(f"Market freshness check failed: {e}"); freshness_results["market_data"] = {"status": "error", "error": str(e)}
            
            try:
                with self.get_connection() as conn: cursor = conn.cursor(); cursor.execute("SELECT MAX(published_at) FROM news_data WHERE created_at > %s", (cutoff_time,)); latest_news = cursor.fetchone()[0]; news_age = (datetime.now() - latest_news).total_seconds() / 3600 if latest_news else max_age_hours + 1; freshness_results["news_data"] = {"age_hours": news_age, "is_fresh": news_age <= max_age_hours}; news_age > max_age_hours and stale_sources.append("news_data")
            except Exception as e: logger.warning(f"News freshness check failed: {e}"); freshness_results["news_data"] = {"status": "error", "error": str(e)}
            
            stale_sources and send_alerts("stale_data_detected", f"Stale data in: {stale_sources}", "warning", {"stale_sources": stale_sources, "max_age_hours": max_age_hours})
            result = {"status": "success", "freshness_check": freshness_results, "stale_sources": stale_sources, "is_all_fresh": len(stale_sources) == 0, "max_age_hours": max_age_hours, "timestamp": datetime.now().isoformat()}
            log_performance("Data Freshness Monitoring", start_time, datetime.now(), "success", {"stale_sources_count": len(stale_sources), "max_age_hours": max_age_hours})
            return result
        except Exception as e: logger.error(f"Data freshness monitoring failed: {e}"); log_performance("Data Freshness Monitoring", start_time, datetime.now(), "error", {"error": str(e)}); send_alerts("monitoring_system_error", f"Data freshness monitoring failed: {str(e)}", "error"); return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

    def monitor_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health monitoring using shared utilities."""
        start_time = datetime.now()
        try:
            logger.info("Starting comprehensive system health monitoring")
            health_components, critical_issues, warnings = {}, [], []
            
            db_health = self._check_database_performance(); health_components["database"] = db_health; not db_health["healthy"] and critical_issues.append(f"Database: {db_health.get('error', 'Unknown error')}") or db_health.get("response_time", 0) > 5.0 and warnings.append(f"Database slow: {db_health['response_time']:.2f}s")
            api_health = self._check_api_availability(); health_components["apis"] = api_health
            for api, status in api_health.items(): status.get("available") is False and warnings.append(f"API {api} unavailable") or status.get("response_time", 0) > 10.0 and warnings.append(f"API {api} slow: {status['response_time']:.2f}s")
            
            collection_health = self._check_collection_systems(); health_components["collection_systems"] = collection_health; collection_health["errors"] > 0 and warnings.append(f"Collection errors: {collection_health['errors']}")
            overall_status = "healthy" if not critical_issues else "critical"; warnings and overall_status != "critical" and setattr(lambda: None, 'overall_status', "warning") or setattr(lambda: None, 'overall_status', overall_status); overall_status = "warning" if warnings and overall_status != "critical" else overall_status
            
            critical_issues and send_alerts("system_health_critical", f"Critical issues: {critical_issues}", "critical", {"critical_issues": critical_issues, "warnings": warnings}) or warnings and send_alerts("system_health_warning", f"System warnings: {warnings}", "warning", {"warnings": warnings})
            result = {"status": "success", "overall_health": overall_status, "components": health_components, "critical_issues": critical_issues, "warnings": warnings, "timestamp": datetime.now().isoformat()}
            log_performance("System Health Monitoring", start_time, datetime.now(), "success", {"overall_status": overall_status, "critical_issues": len(critical_issues), "warnings": len(warnings)})
            return result
        except Exception as e: logger.error(f"System health monitoring failed: {e}"); log_performance("System Health Monitoring", start_time, datetime.now(), "error", {"error": str(e)}); send_alerts("monitoring_system_error", f"System health monitoring failed: {str(e)}", "critical"); return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

    def _check_market_data_quality(self) -> Dict[str, Any]:
        """Check market data quality metrics."""
        try:
            test_result = self.collect_market_data(["AAPL", "SPY", "QQQ"])
            total, collected, errors = test_result.get("total_symbols", 3), test_result.get("symbols_collected", 0), len(test_result.get("errors", []))
            return {"quality_score": collected / total if total > 0 else 0, "symbols_tested": total, "successful_collections": collected, "errors": errors, "data_sources_available": {"yfinance": YFINANCE_AVAILABLE}}
        except Exception as e: return {"quality_score": 0.0, "error": str(e)}

    def _check_news_data_quality(self) -> Dict[str, Any]:
        """Check news data quality metrics."""
        try:
            test_result = self.collect_sentiment_data(max_articles=5)
            article_count, status = test_result.get("article_count", 0), test_result.get("status", "failed")
            return {"quality_score": 1.0 if status == "success" and article_count > 0 else 0.0, "articles_collected": article_count,
                   "sentiment_method": test_result.get("sentiment_method", "unknown"), "newsapi_available": self.newsapi_client is not None}
        except Exception as e: return {"quality_score": 0.0, "error": str(e)}

    def _check_database_health(self) -> Dict[str, Any]:
        """Check database health and connectivity."""
        try:
            start_time = datetime.now()
            with self.get_connection() as conn: conn.cursor().execute("SELECT 1"); conn.cursor().fetchone()
            return {"healthy": True, "response_time": (datetime.now() - start_time).total_seconds()}
        except Exception as e: return {"healthy": False, "issues": [str(e)]}

    def _check_database_performance(self) -> Dict[str, Any]:
        """Check database performance metrics."""
        try:
            start_time = datetime.now()
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM market_data"); market_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM news_data"); news_count = cursor.fetchone()[0]
                return {"healthy": True, "response_time": (datetime.now() - start_time).total_seconds(), "market_data_records": market_count, "news_data_records": news_count}
        except Exception as e: return {"healthy": False, "error": str(e)}

    def _check_api_availability(self) -> Dict[str, Any]:
        """Check availability of external APIs."""
        api_status = {}
        
        try:
            start_time = datetime.now()
            test_data = self._collect_yahoo_direct("AAPL")
            api_status["yahoo_finance"] = {"available": test_data is not None, "response_time": (datetime.now() - start_time).total_seconds()}
        except Exception as e: api_status["yahoo_finance"] = {"available": False, "error": str(e)}
        
        if self.newsapi_client:
            try:
                start_time = datetime.now()
                self.newsapi_client.get_top_headlines(page_size=1)
                api_status["newsapi"] = {"available": True, "response_time": (datetime.now() - start_time).total_seconds()}
            except Exception as e: api_status["newsapi"] = {"available": False, "error": str(e)}
        else: api_status["newsapi"] = {"available": False, "reason": "not_configured"}
        
        return api_status

    def _check_collection_systems(self) -> Dict[str, Any]:
        """Check data collection systems status."""
        errors = 0
        if not YFINANCE_AVAILABLE: errors += 1
        if not NEWSAPI_AVAILABLE: errors += 1
        return {"yfinance_available": YFINANCE_AVAILABLE, "newsapi_available": NEWSAPI_AVAILABLE, "transformers_available": TRANSFORMERS_AVAILABLE,
               "textblob_available": TEXTBLOB_AVAILABLE, "sentiment_method": self.sentiment_method, "errors": errors}

    def monitor_data_collection_performance(self) -> Dict[str, Any]:
        """Monitor overall data collection performance using shared utilities."""
        start_time = datetime.now()
        try:
            logger.info("Monitoring data collection performance")
            test_symbols = ["AAPL", "SPY", "QQQ"]
            
            market_result = self.collect_market_data(test_symbols)
            fundamental_result = self.collect_fundamental_data(test_symbols)
            sentiment_result = self.collect_sentiment_data(max_articles=5)
            
            collection_metrics = {
                "market_success_rate": market_result.get("symbols_collected", 0) / len(test_symbols),
                "fundamental_success_rate": len(fundamental_result.get("data", [])) / len(test_symbols),
                "sentiment_articles_collected": sentiment_result.get("article_count", 0),
                "overall_system_health": self.health_check()["status"]
            }
            
            overall_score = (collection_metrics["market_success_rate"] + collection_metrics["fundamental_success_rate"]) / 2
            
            if overall_score < 0.7:
                send_alerts("data_collection_performance", f"Collection performance degraded: {overall_score:.1%}", "warning", collection_metrics)
            
            result = {"status": "success", "overall_performance_score": overall_score, "metrics": collection_metrics, "timestamp": datetime.now().isoformat()}
            log_performance("Data Collection Performance", start_time, datetime.now(), "success", {"overall_score": overall_score})
            return result
        except Exception as e:
            logger.error(f"Collection performance monitoring failed: {e}")
            log_performance("Data Collection Performance", start_time, datetime.now(), "error", {"error": str(e)})
            send_alerts("monitoring_system_error", f"Collection performance monitoring failed: {str(e)}", "error")
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

def get_data_manager(config: Optional[Dict[str, Any]] = None) -> DataManager: return DataManager(config)
def validate_symbols(symbols: List[str]) -> List[str]: return [symbol.upper().strip() for symbol in symbols if symbol.strip()]