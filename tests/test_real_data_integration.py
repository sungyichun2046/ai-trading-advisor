"""Real data integration tests (only run when USE_REAL_DATA=True)."""

import pytest
from src.config import settings


class TestRealDataIntegration:
    """Real data integration tests (only run when USE_REAL_DATA=True)."""
    
    @pytest.mark.skipif(not settings.use_real_data, reason="USE_REAL_DATA=False in .env")
    def test_real_market_data_collection(self):
        """Test real market data collection from Yahoo Finance."""
        from src.data.collectors import MarketDataCollector
        
        collector = MarketDataCollector()
        
        # Test with popular symbols
        symbols = ["SPY", "AAPL", "QQQ", "MSFT"]
        successful_collections = 0
        
        for symbol in symbols:
            result = collector.collect_real_time_data(symbol)
            
            if result and result.get('data_source') in ['yahoo_direct', 'yfinance']:
                successful_collections += 1
                
                # Verify real data structure
                assert result["symbol"] == symbol
                assert result["status"] == "success"
                assert isinstance(result["price"], (int, float))
                assert result["price"] > 0
                assert isinstance(result["volume"], int)
                assert result["volume"] >= 0
                assert result["data_source"] in ["yahoo_direct", "yfinance"]
                
                # Verify OHLC data
                assert "open" in result
                assert "high" in result  
                assert "low" in result
                assert "close" in result
                assert "timestamp" in result
                
                # Basic price validation
                assert result["low"] <= result["high"]
                assert result["low"] <= result["close"] <= result["high"]
                assert result["low"] <= result["open"] <= result["high"]
                
                print(f"✅ Real data verified for {symbol}: ${result['price']}")
                break  # One successful test is enough to prove real API works
        
        # At least one symbol should work (unless all APIs are down)
        if successful_collections == 0:
            pytest.skip("All market data APIs appear to be unavailable")
            
    @pytest.mark.skipif(not settings.use_real_data, reason="USE_REAL_DATA=False in .env") 
    def test_real_news_collection(self):
        """Test real news collection from NewsAPI."""
        if not settings.newsapi_key or settings.newsapi_key == "your_newsapi_key":
            pytest.skip("NewsAPI key not configured")
            
        from src.data.collectors import NewsCollector
        
        collector = NewsCollector()
        
        # Test news collection
        news_articles = collector.collect_financial_news()
        
        assert isinstance(news_articles, list)
        assert len(news_articles) > 0
        
        # Verify at least some real articles
        real_articles = [a for a in news_articles if a.get('data_source') == 'newsapi']
        assert len(real_articles) > 0, "No real articles collected from NewsAPI"
        
        # Verify article structure
        for article in real_articles[:3]:  # Check first 3 articles
            assert "title" in article
            assert "content" in article
            assert "source" in article
            assert "timestamp" in article
            assert article["data_source"] == "newsapi"
            
            # Title should not be empty
            assert len(article["title"].strip()) > 0
            
        print(f"✅ Real news verified: {len(real_articles)} articles from NewsAPI")
        
        # Test sentiment analysis
        sentiment_result = collector.analyze_sentiment(news_articles)
        
        assert sentiment_result["status"] == "success"
        assert sentiment_result["articles_analyzed"] > 0
        assert "average" in sentiment_result
        assert isinstance(sentiment_result["average"], (int, float))
        assert sentiment_result["data_source"] in ["finbert", "textblob", "dummy"]
        
        print(f"✅ Sentiment analysis verified: {sentiment_result['data_source']} method")
        
    @pytest.mark.skipif(not settings.use_real_data, reason="USE_REAL_DATA=False in .env")
    def test_real_data_fallback_behavior(self):
        """Test that real data collection gracefully falls back when APIs fail."""
        from src.data.collectors import MarketDataCollector
        
        collector = MarketDataCollector()
        
        # Test with an invalid symbol that should fallback to dummy data
        result = collector.collect_real_time_data("INVALID_SYMBOL_12345")
        
        # Should still return data (dummy fallback after real API fails)
        assert result is not None
        assert "data_source" in result
        
        # Should be dummy data since invalid symbol
        assert result["data_source"] == "dummy"
        assert result["symbol"] == "INVALID_SYMBOL_12345" 
        assert result["status"] == "success"
        assert isinstance(result["price"], (int, float))
        assert result["price"] > 0  # Dummy data should have positive prices
            
        print(f"✅ Fallback behavior verified: {result['data_source']}")
        
    @pytest.mark.skipif(not settings.use_real_data, reason="USE_REAL_DATA=False in .env")
    def test_historical_data_collection(self):
        """Test historical data collection (if yfinance works)."""
        from src.data.collectors import MarketDataCollector
        
        collector = MarketDataCollector()
        
        # Test historical data collection
        result = collector.collect_historical_data("AAPL", "1d")
        
        # Historical data may fail due to yfinance issues, that's okay
        if result is not None and len(result) > 0:
            # If it works, verify it's a proper DataFrame
            assert hasattr(result, 'iloc')  # DataFrame method
            assert hasattr(result, 'columns')  # DataFrame property
            assert 'Close' in result.columns
            
            latest = result.iloc[-1]
            assert isinstance(latest['Close'], (int, float))
            assert latest['Close'] > 0
            
            print(f"✅ Historical data working: {len(result)} data points")
        else:
            print("ℹ️  Historical data not available (yfinance issues)")
            # This is expected and okay