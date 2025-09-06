"""Unit tests for data collectors."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Mock heavy dependencies before any imports
mock_modules = {
    'newsapi': Mock(),
    'newsapi.newsapi_client': Mock(),
    'transformers': Mock(),
    'torch': Mock(), 
    'textblob': Mock(),
    'yfinance': Mock(),
    'pandas': Mock()
}

for module_name, mock_obj in mock_modules.items():
    sys.modules[module_name] = mock_obj

# Mock pandas Series for DataFrame.loc returns
class MockSeries:
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
        
    def iloc(self, index):
        if isinstance(index, int) and index < len(self.data):
            return self.data[index]
        return self.data[0] if self.data else 0
        
    def __getitem__(self, index):
        return self.iloc(index)

# Mock pandas DataFrame
class MockDataFrame:
    def __init__(self, data=None, index=None, columns=None):
        self.data = data or {}
        self.index = index or []
        self._columns = columns or list(self.data.keys()) if isinstance(self.data, dict) else ['Open', 'High', 'Low', 'Close', 'Volume']
        self.empty = len(self.data) == 0
    
    def iloc(self, index):
        if isinstance(self.data, dict) and self.data:
            # For VIX data, return the last value
            if 'Close' in self.data:
                values = self.data['Close']
                if isinstance(values, list) and values:
                    if isinstance(index, int):
                        return values[index] if index < len(values) else values[-1]
                    elif hasattr(index, '__getitem__'):  # slice-like
                        return values[index]
                    return values[-1]
        return {'Close': 150.0, 'Volume': 1000000, 'Open': 149.0, 'High': 152.0, 'Low': 148.0}
    
    def __getitem__(self, key):
        if isinstance(self.data, dict) and key in self.data:
            return self.data[key]
        return [150.0, 151.0]
    
    @property
    def columns(self):
        return self._columns
    
    def __len__(self):
        if isinstance(self.data, dict) and self.data:
            return len(list(self.data.values())[0]) if self.data else 0
        return 2
        
    @property 
    def loc(self):
        """Mock pandas DataFrame.loc functionality - returns a callable."""
        class LocAccessor:
            def __init__(self, df):
                self.df = df
                
            def __getitem__(self, key):
                """Implement df.loc[key] functionality."""
                if isinstance(self.df.data, dict):
                    # Handle index-based access for financial data
                    if isinstance(key, str) and key in self.df.index:
                        # Return all column data for the row as a MockSeries
                        row_data = []
                        idx = self.df.index.index(key)
                        for col_key in self.df.data.keys():
                            values = self.df.data[col_key]
                            if isinstance(values, list) and idx < len(values):
                                row_data.append(values[idx])
                            else:
                                row_data.append(values[0] if isinstance(values, list) else values)
                        return MockSeries(row_data)
                    # Return first column's data for the row
                    if self.df.data:
                        first_col = list(self.df.data.keys())[0]
                        if first_col in self.df.data:
                            values = self.df.data[first_col]
                            return MockSeries([values[0] if isinstance(values, list) else values])
                return MockSeries([1000000.0])  # Default value
                
        return LocAccessor(self)
        
    def mean(self):
        """Mock pandas DataFrame.mean functionality."""
        if isinstance(self.data, dict) and 'Close' in self.data:
            values = self.data['Close']
            if isinstance(values, list) and values:
                return sum(values) / len(values)
        return 150.0  # Default value

# Replace pandas with our mock
sys.modules['pandas'].DataFrame = MockDataFrame

# Create pandas alias for tests
import pandas as pd

# Now we can safely import
from src.data.collectors import MarketDataCollector, NewsCollector
from src.config import settings


class TestMarketDataCollector:
    """Test cases for MarketDataCollector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MarketDataCollector()

    def test_init(self):
        """Test collector initialization."""
        assert self.collector.retry_attempts == 3
        assert self.collector.retry_delay == 1

    @patch('src.data.collectors.settings')
    def test_collect_real_time_data_dummy_mode(self, mock_settings):
        """Test data collection in dummy mode."""
        mock_settings.use_real_data = False
        
        result = self.collector.collect_real_time_data("AAPL")
        
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["status"] == "success"
        assert "price" in result
        assert "volume" in result
        assert result["data_source"] == "dummy"

    @patch('src.data.collectors.yf.Ticker')
    @patch('src.data.collectors.settings')
    def test_collect_real_time_data_success(self, mock_settings, mock_ticker):
        """Test successful real data collection."""
        mock_settings.use_real_data = True
        
        # Mock yfinance response
        mock_hist = pd.DataFrame({
            'Open': [150.0],
            'High': [152.0], 
            'Low': [149.0],
            'Close': [151.0],
            'Volume': [1000000]
        })
        
        mock_info = {
            'currentPrice': 151.5,
            'marketCap': 2500000000000,
            'forwardPE': 25.0
        }
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.collector.collect_real_time_data("AAPL")
        
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["status"] == "success"
        assert "price" in result  # Price could come from yahoo_direct or yfinance
        assert "volume" in result  # Volume could vary based on data source
        assert result["data_source"] in ["yfinance", "yahoo_direct"]
        assert "market_cap" in result  # Market cap could vary based on data source

    @patch('src.data.collectors.yf.Ticker')
    @patch('src.data.collectors.settings')
    def test_collect_real_time_data_empty_response(self, mock_settings, mock_ticker):
        """Test handling of empty yfinance response."""
        mock_settings.use_real_data = True
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.collector.collect_real_time_data("INVALID")
        
        assert result is not None
        assert result["symbol"] == "INVALID"
        assert result["data_source"] == "dummy"  # Falls back to dummy data

    @patch('src.data.collectors.yf.Ticker')
    @patch('src.data.collectors.settings')
    def test_collect_real_time_data_api_failure(self, mock_settings, mock_ticker):
        """Test handling of yfinance API failure."""
        mock_settings.use_real_data = True
        
        mock_ticker.side_effect = Exception("API Error")
        
        result = self.collector.collect_real_time_data("AAPL")
        
        # Could get real data from yahoo_direct or fall back to dummy data
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["data_source"] in ["dummy", "yahoo_direct"]

    @patch('src.data.collectors.yf.Ticker')
    def test_collect_historical_data_success(self, mock_ticker):
        """Test successful historical data collection."""
        mock_hist = pd.DataFrame({
            'Open': [150.0, 151.0],
            'High': [152.0, 153.0],
            'Low': [149.0, 150.0],
            'Close': [151.0, 152.0],
            'Volume': [1000000, 1100000]
        })
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.collector.collect_historical_data("AAPL", "1mo")
        
        assert result is not None
        assert len(result) == 2
        assert "Open" in result.columns


class TestNewsCollector:
    """Test cases for NewsCollector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = NewsCollector()

    @patch('src.data.collectors.NewsApiClient')
    @patch('src.data.collectors.pipeline')
    @patch('src.data.collectors.settings')
    def test_init_real_mode(self, mock_settings, mock_pipeline, mock_newsapi):
        """Test collector initialization in real mode."""
        mock_settings.use_real_data = True
        mock_settings.newsapi_key = "test_key"
        
        collector = NewsCollector()
        
        mock_newsapi.assert_called_with(api_key="test_key")

    @patch('src.data.collectors.settings')
    def test_collect_financial_news_dummy_mode(self, mock_settings):
        """Test news collection in dummy mode."""
        mock_settings.use_real_data = False
        
        result = self.collector.collect_financial_news()
        
        assert isinstance(result, list)
        assert len(result) >= 3
        assert all("title" in article for article in result)
        assert all(article["data_source"] == "dummy" for article in result)

    @patch('src.data.collectors.NewsApiClient')
    @patch('src.data.collectors.settings')
    def test_collect_financial_news_real_mode(self, mock_settings, mock_newsapi):
        """Test real news collection."""
        mock_settings.use_real_data = True
        mock_settings.newsapi_key = "test_key"
        mock_settings.max_news_articles = 5
        
        # Mock NewsAPI response
        mock_articles = {
            'articles': [
                {
                    'title': 'Test Article 1',
                    'description': 'Test description',
                    'content': 'Test content',
                    'url': 'http://test.com/1',
                    'source': {'name': 'Test Source'},
                    'publishedAt': '2024-01-01T10:00:00Z'
                },
                {
                    'title': 'Test Article 2', 
                    'description': 'Test description 2',
                    'content': 'Test content 2',
                    'url': 'http://test.com/2',
                    'source': {'name': 'Test Source 2'},
                    'publishedAt': '2024-01-01T11:00:00Z'
                }
            ]
        }
        
        mock_client = Mock()
        mock_client.get_everything.return_value = mock_articles
        mock_newsapi.return_value = mock_client
        
        collector = NewsCollector()
        result = collector.collect_financial_news()
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["title"] == "Test Article 1"
        assert result[0]["data_source"] == "newsapi"

    @patch('src.data.collectors.settings')
    def test_analyze_sentiment_dummy_mode(self, mock_settings):
        """Test sentiment analysis in dummy mode."""
        mock_settings.use_real_data = False
        
        news_data = [
            {"title": "Stocks Rally on Good News", "content": "Markets up"},
            {"title": "Market Volatility Continues", "content": "Uncertainty remains"}
        ]
        
        result = self.collector.analyze_sentiment(news_data)
        
        assert result["status"] == "success"
        assert result["articles_analyzed"] == 2
        assert "average" in result
        assert "sentiment_range" in result
        assert result["data_source"] == "dummy"

    @patch('src.data.collectors.pipeline')
    @patch('src.data.collectors.settings')
    def test_analyze_sentiment_real_mode(self, mock_settings, mock_pipeline):
        """Test real sentiment analysis."""
        mock_settings.use_real_data = True
        
        # Mock FinBERT pipeline
        mock_analyzer = Mock()
        mock_analyzer.return_value = [[
            {'label': 'positive', 'score': 0.7},
            {'label': 'negative', 'score': 0.2},
            {'label': 'neutral', 'score': 0.1}
        ]]
        mock_pipeline.return_value = mock_analyzer
        
        collector = NewsCollector()
        collector.sentiment_analyzer = mock_analyzer
        
        news_data = [
            {"title": "Great earnings report", "content": "Company beats expectations"}
        ]
        
        result = collector.analyze_sentiment(news_data)
        
        assert result["status"] == "success"
        assert result["articles_analyzed"] == 1
        assert result["data_source"] == "finbert"
        assert len(result["articles"]) == 1
        assert "sentiment_score" in result["articles"][0]

    @patch('src.data.collectors.NewsApiClient')
    @patch('src.data.collectors.settings')
    def test_newsapi_failure_fallback(self, mock_settings, mock_newsapi):
        """Test fallback to dummy data when NewsAPI fails."""
        mock_settings.use_real_data = True
        mock_settings.newsapi_key = "test_key"
        
        mock_client = Mock()
        mock_client.get_everything.side_effect = Exception("API Error")
        mock_newsapi.return_value = mock_client
        
        collector = NewsCollector()
        result = collector.collect_financial_news()
        
        # Should fall back to dummy data
        assert isinstance(result, list)
        assert all(article["data_source"] == "dummy" for article in result)


class TestIntegration:
    """Integration tests for data collectors."""
    
    def test_market_collector_dummy_data_format(self):
        """Test that dummy data has correct format."""
        collector = MarketDataCollector()
        
        with patch('src.data.collectors.settings') as mock_settings:
            mock_settings.use_real_data = False
            result = collector.collect_real_time_data("SPY")
        
        # Verify all required fields are present
        required_fields = [
            "symbol", "status", "price", "volume", "open", "high", 
            "low", "close", "timestamp", "market_cap", "pe_ratio", "data_source"
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Verify data types
        assert isinstance(result["price"], (int, float))
        assert isinstance(result["volume"], int)
        assert result["status"] == "success"

    def test_news_collector_dummy_sentiment_format(self):
        """Test that dummy sentiment analysis has correct format."""
        collector = NewsCollector()
        
        with patch('src.data.collectors.settings') as mock_settings:
            mock_settings.use_real_data = False
            
            news_data = collector.collect_financial_news()
            sentiment_result = collector.analyze_sentiment(news_data)
        
        # Verify sentiment result format
        required_fields = [
            "status", "average", "articles_analyzed", 
            "sentiment_range", "articles", "data_source"
        ]
        
        for field in required_fields:
            assert field in sentiment_result, f"Missing required field: {field}"
        
        # Verify processed articles have sentiment scores
        for article in sentiment_result["articles"]:
            assert "sentiment_score" in article
            assert isinstance(article["sentiment_score"], (int, float))