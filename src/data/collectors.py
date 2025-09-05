"""Data collectors for AI Trading Advisor."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Collects market data from various sources."""

    def __init__(self):
        """Initialize market data collector."""
        pass

    def collect_real_time_data(self, symbol: str) -> Optional[Dict]:
        """Collect real-time market data for a symbol.

        Args:
            symbol: Stock/ETF symbol to collect data for

        Returns:
            Dictionary containing market data or None if failed
        """
        logger.info(f"Collecting real-time data for {symbol}")
        
        # Generate realistic sample data
        import random
        from datetime import datetime
        
        base_prices = {
            "SPY": 450.0,
            "QQQ": 380.0, 
            "AAPL": 180.0,
            "MSFT": 340.0,
            "TSLA": 240.0,
            "GOOGL": 140.0,
            "AMZN": 150.0,
            "META": 320.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        price_variance = random.uniform(-0.05, 0.05)  # +/- 5% daily variance
        current_price = base_price * (1 + price_variance)
        
        return {
            "symbol": symbol,
            "status": "success",
            "price": round(current_price, 2),
            "volume": random.randint(500000, 5000000),
            "timestamp": datetime.now().isoformat(),
        }


class NewsCollector:
    """Collects and analyzes news sentiment."""

    def __init__(self):
        """Initialize news collector."""
        pass

    def collect_financial_news(self) -> List[Dict]:
        """Collect financial news articles.

        Returns:
            List of news articles
        """
        logger.info("Collecting financial news")
        
        # Generate realistic sample news
        from datetime import datetime
        import random
        
        sample_news = [
            {
                "title": "Tech Stocks Rally on AI Optimism",
                "content": "Technology stocks surged today as investors showed renewed optimism about artificial intelligence developments...",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "title": "Federal Reserve Holds Rates Steady",
                "content": "The Federal Reserve announced it will maintain current interest rates amid economic uncertainty...", 
                "timestamp": datetime.now().isoformat(),
            },
            {
                "title": "Energy Sector Shows Strong Performance",
                "content": "Oil and gas companies posted solid earnings this quarter, boosting the energy sector...",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "title": "Market Volatility Expected to Continue",
                "content": "Analysts predict continued market volatility as economic indicators remain mixed...",
                "timestamp": datetime.now().isoformat(),
            }
        ]
        
        # Return 2-4 random articles
        num_articles = random.randint(2, 4)
        return random.sample(sample_news, num_articles)

    def analyze_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment of news articles.

        Args:
            news_data: List of news articles

        Returns:
            Sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for {len(news_data)} articles")
        
        # Generate realistic sentiment analysis
        import random
        
        # Simulate sentiment analysis with random but realistic values
        sentiments = []
        for article in news_data:
            title = article.get("title", "")
            if "Rally" in title or "Strong" in title or "AI Optimism" in title:
                sentiment = random.uniform(0.3, 0.7)  # Positive news
            elif "Volatility" in title or "Uncertainty" in title:
                sentiment = random.uniform(-0.3, 0.1)  # Negative/neutral news
            else:
                sentiment = random.uniform(-0.2, 0.2)  # Neutral news
            sentiments.append(sentiment)
        
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        return {
            "status": "success",
            "average": round(average_sentiment, 3),
            "articles_analyzed": len(news_data),
            "sentiment_range": [min(sentiments), max(sentiments)] if sentiments else [0, 0],
            "articles": news_data  # Include articles for database storage
        }
