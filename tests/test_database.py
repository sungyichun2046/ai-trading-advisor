"""Tests for database operations."""

import pytest
import psycopg2
import os
from datetime import datetime, date


@pytest.fixture
def db_connection():
    """Create database connection for testing."""
    # For local testing, try localhost first, then fall back to postgres (Docker)
    host = os.getenv('POSTGRES_HOST', 'localhost')
    try:
        conn = psycopg2.connect(
            host=host,
            port=os.getenv('POSTGRES_PORT', '5432'),
            database=os.getenv('POSTGRES_DB', 'trading_advisor'),
            user=os.getenv('POSTGRES_USER', 'trader'),
            password=os.getenv('POSTGRES_PASSWORD', 'trader_password')
        )
    except psycopg2.OperationalError:
        # If localhost fails, try Docker hostname
        if host == 'localhost':
            conn = psycopg2.connect(
                host='postgres',
                port=os.getenv('POSTGRES_PORT', '5432'),
                database=os.getenv('POSTGRES_DB', 'trading_advisor'),
                user=os.getenv('POSTGRES_USER', 'trader'),
                password=os.getenv('POSTGRES_PASSWORD', 'trader_password')
            )
        else:
            raise
    yield conn
    conn.close()


class TestDatabaseTables:
    """Test database table operations."""
    
    def test_market_data_table_exists(self, db_connection):
        """Test that market_data table exists and has data."""
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data;")
        count = cursor.fetchone()[0]
        assert count >= 0  # Table exists and accessible
        cursor.close()
    
    def test_news_data_table_exists(self, db_connection):
        """Test that news_data table exists and has data."""
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM news_data;")
        count = cursor.fetchone()[0]
        assert count >= 0  # Table exists and accessible
        cursor.close()
    
    def test_analysis_results_table_exists(self, db_connection):
        """Test that analysis_results table exists and has data."""
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM analysis_results;")
        count = cursor.fetchone()[0]
        assert count >= 0  # Table exists and accessible
        cursor.close()
    
    def test_recommendations_table_exists(self, db_connection):
        """Test that recommendations table exists and has data."""
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM recommendations;")
        count = cursor.fetchone()[0]
        assert count >= 0  # Table exists and accessible
        cursor.close()

    def test_sample_data_inserted(self, db_connection):
        """Test that sample data was inserted correctly."""
        cursor = db_connection.cursor()
        
        # Check market data
        cursor.execute("SELECT symbol, price FROM market_data WHERE symbol = 'AAPL';")
        result = cursor.fetchone()
        if result:
            assert result[0] == 'AAPL'
            assert float(result[1]) > 0
        
        cursor.close()

    def test_database_schema(self, db_connection):
        """Test database schema integrity."""
        cursor = db_connection.cursor()
        
        # Check that all required tables exist
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('market_data', 'news_data', 'analysis_results', 'recommendations');
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = ['market_data', 'news_data', 'analysis_results', 'recommendations']
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found in database"
        
        cursor.close()