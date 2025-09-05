#!/usr/bin/env python3
"""Update market data with current real prices."""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.collectors import MarketDataCollector

def update_market_prices():
    """Update database with fresh market prices."""
    print("🚀 Updating Market Prices with Real Data")
    print("=" * 50)
    
    try:
        # Override connection parameters for localhost  
        os.environ['POSTGRES_HOST'] = 'localhost'
        os.environ['USE_REAL_DATA'] = 'True'  # Enable real data collection
        
        # Import database manager after setting env vars
        from data.database import DatabaseManager
        
        collector = MarketDataCollector()
        db_manager = DatabaseManager()
        symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA", "META", "NVDA", "QQQ"]
        
        print(f"📈 Collecting fresh data for {len(symbols)} symbols...")
        
        # Collect fresh data
        fresh_data = {}
        for symbol in symbols:
            result = collector.collect_real_time_data(symbol)
            
            if result and result.get("status") == "success":
                fresh_data[symbol] = result
                print(f"  ✅ {symbol}: ${result['price']:.2f} ({result.get('data_source', 'unknown')})")
            else:
                print(f"  ❌ {symbol}: Failed")
        
        if not fresh_data:
            print("❌ No fresh data collected!")
            return False
        
        print(f"\n💾 Updating database...")
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear old data
            cursor.execute("DELETE FROM market_data;")
            print(f"  🗑️  Cleared old data")
            
            # Insert fresh data using the correct column names
            execution_date = datetime.now()
            count = 0
            
            for symbol, data in fresh_data.items():
                cursor.execute("""
                    INSERT INTO market_data (symbol, price, volume, timestamp, source, execution_date) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    symbol,
                    data.get("price", 0),
                    data.get("volume", 0), 
                    data.get("timestamp", execution_date.isoformat()),
                    data.get("data_source", "yahoo_direct"),
                    execution_date.date()
                ))
                count += 1
            
            conn.commit()
            print(f"  ✅ Inserted {count} records")
        
        print(f"\n🎉 Database updated successfully!")
        print(f"\n📋 Verification commands:")
        print(f"  make db-connect")
        print(f"  SELECT symbol, price, source FROM market_data ORDER BY symbol;")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = update_market_prices()
    if not success:
        sys.exit(1)