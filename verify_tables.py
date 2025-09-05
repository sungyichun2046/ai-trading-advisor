#!/usr/bin/env python3
"""Verify database tables and show sample data."""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.database import DatabaseManager

def verify_tables():
    """Verify all tables exist and show their structure."""
    print("🔍 Verifying database tables...")
    
    try:
        # Override connection parameters for localhost
        os.environ['POSTGRES_HOST'] = 'localhost'
        
        db_manager = DatabaseManager()
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            
            print(f"\n📋 Found {len(tables)} tables:")
            for table in tables:
                table_name = table[0]
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cursor.fetchone()[0]
                
                print(f"  ✅ {table_name:<20} ({count:>4} rows)")
            
            # Check new strategic tables specifically
            strategic_tables = ['fundamental_data', 'volatility_events', 'alerts']
            
            print(f"\n🎯 Strategic monitoring tables:")
            for table_name in strategic_tables:
                try:
                    cursor.execute(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' 
                        ORDER BY ordinal_position 
                        LIMIT 5;
                    """)
                    columns = cursor.fetchall()
                    
                    print(f"  ✅ {table_name}:")
                    for col_name, col_type in columns:
                        print(f"     - {col_name}: {col_type}")
                    
                except Exception as e:
                        print(f"  ❌ {table_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying tables: {e}")
        return False

if __name__ == "__main__":
    verify_tables()