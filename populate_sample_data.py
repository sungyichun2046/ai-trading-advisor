#!/usr/bin/env python3
"""Populate sample data for AI Trading Advisor."""

import os
import psycopg2
import psycopg2.extras
import json
from datetime import datetime, date

def get_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'trading_advisor'),
        user=os.getenv('POSTGRES_USER', 'trader'),
        password=os.getenv('POSTGRES_PASSWORD', 'trader_password')
    )

def populate_market_data(conn):
    """Populate sample market data."""
    cursor = conn.cursor()
    
    market_data = [
        ('AAPL', 175.50, 1000000, datetime.now(), date.today()),
        ('GOOGL', 2750.30, 500000, datetime.now(), date.today()),
        ('MSFT', 380.75, 750000, datetime.now(), date.today()),
        ('TSLA', 220.45, 2000000, datetime.now(), date.today()),
        ('AMZN', 145.80, 800000, datetime.now(), date.today())
    ]
    
    for symbol, price, volume, timestamp, execution_date in market_data:
        cursor.execute("""
            INSERT INTO market_data (symbol, price, volume, timestamp, execution_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (symbol, price, volume, timestamp, execution_date))
    
    conn.commit()
    print(f"Inserted {len(market_data)} market data records")

def populate_news_data(conn):
    """Populate sample news data."""
    cursor = conn.cursor()
    
    news_data = [
        ('Apple Reports Strong Q4 Earnings', 'Apple Inc. reported strong quarterly earnings...', 0.75, datetime.now(), date.today()),
        ('Google AI Breakthrough Announced', 'Google announces major AI breakthrough...', 0.60, datetime.now(), date.today()),
        ('Microsoft Cloud Growth Continues', 'Microsoft Azure continues strong growth...', 0.55, datetime.now(), date.today()),
        ('Tesla Production Targets Met', 'Tesla meets production targets for the quarter...', 0.45, datetime.now(), date.today()),
        ('Amazon Prime Day Success', 'Amazon Prime Day breaks sales records...', 0.65, datetime.now(), date.today())
    ]
    
    for title, content, sentiment, timestamp, execution_date in news_data:
        cursor.execute("""
            INSERT INTO news_data (title, content, sentiment, timestamp, execution_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (title, content, sentiment, timestamp, execution_date))
    
    conn.commit()
    print(f"Inserted {len(news_data)} news data records")

def populate_analysis_results(conn):
    """Populate sample analysis results."""
    cursor = conn.cursor()
    
    analysis_results = [
        ('AAPL', 'technical', {'rsi': 65.5, 'macd': 'bullish', 'sma_20': 172.30}, datetime.now(), date.today()),
        ('GOOGL', 'technical', {'rsi': 58.2, 'macd': 'neutral', 'sma_20': 2745.80}, datetime.now(), date.today()),
        ('MSFT', 'fundamental', {'pe_ratio': 28.5, 'revenue_growth': 0.15, 'debt_ratio': 0.25}, datetime.now(), date.today()),
        ('TSLA', 'sentiment', {'overall_sentiment': 0.45, 'news_count': 15, 'social_sentiment': 0.35}, datetime.now(), date.today()),
        ('AMZN', 'risk', {'volatility': 0.28, 'beta': 1.15, 'var_95': 0.045}, datetime.now(), date.today())
    ]
    
    for symbol, analysis_type, results, timestamp, execution_date in analysis_results:
        cursor.execute("""
            INSERT INTO analysis_results (symbol, analysis_type, results, timestamp, execution_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (symbol, analysis_type, psycopg2.extras.Json(results), timestamp, execution_date))
    
    conn.commit()
    print(f"Inserted {len(analysis_results)} analysis results records")

def populate_recommendations(conn):
    """Populate sample recommendations."""
    cursor = conn.cursor()
    
    recommendations = [
        ('AAPL', 'BUY', 0.85, 1000.00, 'MEDIUM', datetime.now(), date.today()),
        ('GOOGL', 'HOLD', 0.60, 0.00, 'LOW', datetime.now(), date.today()),
        ('MSFT', 'BUY', 0.75, 1500.00, 'MEDIUM', datetime.now(), date.today()),
        ('TSLA', 'SELL', 0.70, 500.00, 'HIGH', datetime.now(), date.today()),
        ('AMZN', 'HOLD', 0.55, 0.00, 'LOW', datetime.now(), date.today())
    ]
    
    for symbol, action, confidence, position_size, risk_level, timestamp, execution_date in recommendations:
        cursor.execute("""
            INSERT INTO recommendations (symbol, action, confidence, position_size, risk_level, timestamp, execution_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (symbol, action, confidence, position_size, risk_level, timestamp, execution_date))
    
    conn.commit()
    print(f"Inserted {len(recommendations)} recommendation records")

def main():
    """Main function to populate all sample data."""
    print("Populating sample data for AI Trading Advisor...")
    
    try:
        conn = get_connection()
        print("Connected to database successfully")
        
        populate_market_data(conn)
        populate_news_data(conn)
        populate_analysis_results(conn)
        populate_recommendations(conn)
        
        print("Sample data population completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure PostgreSQL is running and accessible")
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()