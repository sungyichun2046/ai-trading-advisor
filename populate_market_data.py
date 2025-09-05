#!/usr/bin/env python3
"""Script to populate database with real market data."""

import sys
import os
import psycopg2
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.collectors import MarketDataCollector, NewsCollector
from src.config import settings

def create_tables():
    """Create database tables if they don't exist."""
    try:
        # Database connection
        conn = psycopg2.connect(
            host="localhost",
            database="trading_advisor", 
            user="trader",
            password="trader_password",
            port=5432
        )
        
        cur = conn.cursor()
        
        # Create market_data table (matching Makefile schema)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY, 
                symbol VARCHAR(10), 
                price DECIMAL, 
                volume BIGINT, 
                timestamp TIMESTAMP, 
                execution_date DATE
            );
        """)
        
        # Create news_data table (matching Makefile schema)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_data (
                id SERIAL PRIMARY KEY, 
                title TEXT, 
                content TEXT, 
                sentiment DECIMAL, 
                timestamp TIMESTAMP, 
                execution_date DATE
            );
        """)
        
        conn.commit()
        print("✅ Database tables created successfully")
        return conn
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def populate_market_data(conn):
    """Populate market data table with real data."""
    collector = MarketDataCollector()
    symbols = ["SPY", "AAPL", "MSFT", "QQQ", "GOOGL", "AMZN", "TSLA", "META"]
    
    cur = conn.cursor()
    successful_inserts = 0
    
    for symbol in symbols:
        print(f"Collecting data for {symbol}...")
        
        try:
            data = collector.collect_real_time_data(symbol)
            
            if data and data.get('status') == 'success':
                # Insert into database (matching simple schema)
                cur.execute("""
                    INSERT INTO market_data 
                    (symbol, price, volume, timestamp, execution_date)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    data['symbol'],
                    data['price'],
                    data['volume'],
                    datetime.now(),
                    datetime.now().date()
                ))
                
                successful_inserts += 1
                print(f"  ✅ {symbol}: ${data['price']} ({data['data_source']})")
            else:
                print(f"  ❌ {symbol}: Failed to collect data")
                
        except Exception as e:
            print(f"  ❌ {symbol}: Error - {e}")
    
    conn.commit()
    print(f"\n✅ Successfully inserted {successful_inserts} market data records")
    return successful_inserts

def populate_news_data(conn):
    """Populate news data table with real data."""
    collector = NewsCollector()
    
    print("Collecting financial news...")
    
    try:
        # Collect news articles
        news_articles = collector.collect_financial_news()
        
        if not news_articles:
            print("❌ No news articles collected")
            return 0
        
        # Analyze sentiment
        sentiment_result = collector.analyze_sentiment(news_articles)
        
        if sentiment_result.get('status') != 'success':
            print("❌ Sentiment analysis failed")
            return 0
        
        # Insert articles with sentiment scores
        cur = conn.cursor()
        successful_inserts = 0
        
        for article in sentiment_result.get('articles', []):
            try:
                cur.execute("""
                    INSERT INTO news_data 
                    (title, content, sentiment, timestamp, execution_date)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    article.get('title', '')[:200],  # Limit title length
                    article.get('content', '')[:500],  # Limit content length
                    article.get('sentiment_score', 0),
                    datetime.now(),
                    datetime.now().date()
                ))
                successful_inserts += 1
                
            except Exception as e:
                print(f"  ❌ Failed to insert article: {e}")
        
        conn.commit()
        print(f"✅ Successfully inserted {successful_inserts} news articles")
        print(f"📊 Average sentiment: {sentiment_result.get('average', 0):.3f}")
        
        return successful_inserts
        
    except Exception as e:
        print(f"❌ News collection failed: {e}")
        return 0

def show_data(conn):
    """Show sample data from tables."""
    cur = conn.cursor()
    
    print("\n" + "="*60)
    print("📊 MARKET DATA SAMPLE")
    print("="*60)
    
    cur.execute("""
        SELECT symbol, price, volume, timestamp 
        FROM market_data 
        ORDER BY timestamp DESC 
        LIMIT 5
    """)
    
    rows = cur.fetchall()
    if rows:
        for row in rows:
            symbol, price, volume, timestamp = row
            print(f"{symbol:6} | ${price:8.2f} | {volume:12,} | {timestamp}")
    else:
        print("No market data found")
    
    print("\n" + "="*60)
    print("📰 NEWS DATA SAMPLE")
    print("="*60)
    
    cur.execute("""
        SELECT title, sentiment, timestamp 
        FROM news_data 
        ORDER BY timestamp DESC 
        LIMIT 3
    """)
    
    rows = cur.fetchall()
    if rows:
        for row in rows:
            title, sentiment, timestamp = row
            print(f"📰 {title[:50]}...")
            print(f"   📊 Sentiment: {sentiment:.3f} | {timestamp}")
            print()
    else:
        print("No news data found")

if __name__ == "__main__":
    print("🚀 AI Trading Advisor - Database Population")
    print("="*60)
    
    # Create database connection and tables
    conn = create_tables()
    if not conn:
        exit(1)
    
    try:
        # Populate market data
        market_count = populate_market_data(conn)
        
        # Populate news data
        news_count = populate_news_data(conn)
        
        # Show sample data
        if market_count > 0 or news_count > 0:
            show_data(conn)
        
        print("\n✅ Database population completed!")
        
    finally:
        conn.close()