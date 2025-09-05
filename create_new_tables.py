#!/usr/bin/env python3
"""Create new database tables for strategic data monitoring."""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.database import DatabaseManager

def create_new_tables():
    """Create the new database tables."""
    print("🔧 Creating new database tables...")
    
    try:
        # Override connection parameters for localhost
        import os
        os.environ['POSTGRES_HOST'] = 'localhost'
        
        db_manager = DatabaseManager()
        db_manager.create_tables()
        print("✅ Successfully created all database tables!")
        
        # List tables to verify
        print("\n📋 Verifying tables exist:")
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            
            for table in tables:
                print(f"  ✓ {table[0]}")
        
        # Check specifically for new tables
        expected_new_tables = ['fundamental_data', 'volatility_events', 'alerts']
        existing_tables = [table[0] for table in tables]
        
        print(f"\n🔍 New strategic monitoring tables:")
        for table in expected_new_tables:
            if table in existing_tables:
                print(f"  ✅ {table} - CREATED")
            else:
                print(f"  ❌ {table} - MISSING")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

if __name__ == "__main__":
    success = create_new_tables()
    if success:
        print(f"\n🎉 Database schema updated successfully!")
        print(f"You can now verify with: make db-connect")
        print(f"Then run: \\dt")
    else:
        print(f"\n💥 Failed to update database schema.")
        sys.exit(1)