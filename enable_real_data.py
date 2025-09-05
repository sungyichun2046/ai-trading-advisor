#!/usr/bin/env python3
"""
Script to enable real data collection and test API connectivity.

This script shows how to:
1. Enable real data collection in the configuration
2. Test yfinance connectivity (no API key required)
3. Test API keys for NewsAPI, Alpha Vantage, Polygon (optional)
4. Verify data collection works with real APIs
"""

import os
import sys
sys.path.append('src')

from src.config import settings
from src.data.collectors import MarketDataCollector, NewsCollector, VolatilityMonitor, FundamentalDataCollector
from datetime import datetime

def test_yfinance_connectivity():
    """Test yfinance connection (no API key required)."""
    print("🔍 Testing yfinance connectivity...")
    try:
        import yfinance as yf
        
        # Test basic ticker fetch
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1d", interval="1h")
        
        if not hist.empty:
            print(f"✅ yfinance working - got {len(hist)} data points for AAPL")
            print(f"   Latest price: ${hist['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ yfinance returned empty data")
            return False
            
    except ImportError:
        print("❌ yfinance not installed. Install with: pip install yfinance")
        return False
    except Exception as e:
        print(f"❌ yfinance error: {e}")
        return False

def test_market_data_collector():
    """Test MarketDataCollector with real data."""
    print("\n📊 Testing MarketDataCollector...")
    
    # Enable real data for testing
    settings.use_real_data = True
    
    collector = MarketDataCollector()
    
    # Test collecting real-time data
    symbols = ["AAPL", "SPY", "TSLA"]
    for symbol in symbols:
        try:
            data = collector.collect_real_time_data(symbol)
            if data and data.get("status") == "success":
                print(f"✅ {symbol}: ${data.get('price', 'N/A'):.2f} (Volume: {data.get('volume', 'N/A'):,})")
            else:
                print(f"❌ {symbol}: Failed to collect data")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

def test_volatility_monitor():
    """Test VolatilityMonitor with real data."""
    print("\n📈 Testing VolatilityMonitor...")
    
    settings.use_real_data = True
    
    monitor = VolatilityMonitor()
    
    try:
        result = monitor.check_market_volatility()
        if result and result.get("status") == "success":
            print(f"✅ Volatility check successful")
            print(f"   VIX Current: {result.get('vix_current', 'N/A')}")
            print(f"   Volatility Level: {result.get('volatility_level', 'N/A')}")
            print(f"   Data Source: {result.get('data_source', 'N/A')}")
        else:
            print(f"❌ Volatility check failed")
    except Exception as e:
        print(f"❌ Volatility monitor error: {e}")

def test_fundamental_collector():
    """Test FundamentalDataCollector with real data."""
    print("\n📋 Testing FundamentalDataCollector...")
    
    settings.use_real_data = True
    
    collector = FundamentalDataCollector()
    
    try:
        result = collector.collect_weekly_fundamentals("AAPL")
        if result and result.get("status") == "success":
            print(f"✅ Fundamental data collected for AAPL")
            print(f"   Market Cap: ${result.get('market_cap', 'N/A'):,}")
            print(f"   P/E Ratio: {result.get('pe_ratio', 'N/A')}")
            print(f"   Data Source: {result.get('data_source', 'N/A')}")
        else:
            print(f"❌ Fundamental collection failed")
    except Exception as e:
        print(f"❌ Fundamental collector error: {e}")

def test_news_collector():
    """Test NewsCollector (requires API key)."""
    print("\n📰 Testing NewsCollector...")
    
    settings.use_real_data = True
    
    if not settings.newsapi_key:
        print("⚠️  NewsAPI key not configured. Set NEWSAPI_KEY environment variable to test.")
        print("   Get free API key at: https://newsapi.org/")
        return
    
    collector = NewsCollector()
    
    try:
        result = collector.collect_financial_news()
        if result and result.get("status") == "success":
            articles = result.get("articles", [])
            print(f"✅ News collection successful - {len(articles)} articles")
        else:
            print(f"❌ News collection failed")
    except Exception as e:
        print(f"❌ News collector error: {e}")

def main():
    """Run all connectivity tests."""
    print("🚀 AI Trading Advisor - Real Data Testing")
    print("=" * 50)
    
    # Test basic connectivity
    yfinance_ok = test_yfinance_connectivity()
    
    if not yfinance_ok:
        print("\n❌ yfinance is required for real data collection.")
        print("Install with: pip install yfinance")
        return
    
    # Test collectors
    test_market_data_collector()
    test_volatility_monitor()  
    test_fundamental_collector()
    test_news_collector()
    
    print("\n" + "=" * 50)
    print("🎯 Real Data Testing Complete!")
    print("\nTo enable real data in DAGs:")
    print("1. Set USE_REAL_DATA=True in environment")
    print("2. Restart Airflow services: docker-compose restart airflow-scheduler airflow-webserver")
    print("3. Trigger DAGs through Airflow UI: http://localhost:8080")
    
    print("\n📋 Optional API Keys (for enhanced functionality):")
    print("- NEWSAPI_KEY: For real news sentiment analysis")
    print("- ALPHA_VANTAGE_API_KEY: For additional financial data")
    print("- POLYGON_API_KEY: For high-frequency data")

if __name__ == "__main__":
    main()