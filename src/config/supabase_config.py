"""
Supabase Database Configuration for Multi-Profile Trading Simulator
=================================================================

This module provides configuration and connection management for the Supabase 
PostgreSQL database used in the trading simulator. It includes connection 
pooling, environment-based configuration, and helper utilities.

Features:
- Environment-based configuration (development, staging, production)
- Connection pooling for optimal performance
- SSL configuration for secure connections
- Health check utilities
- Storage monitoring for 500MB limit compliance
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse
import asyncpg
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SupabaseConfig:
    """Configuration class for Supabase database connection."""
    
    # Database Connection
    url: str
    host: str
    port: int
    database: str
    username: str
    password: str
    
    # Connection Pool Settings
    min_connections: int = 1
    max_connections: int = 10
    
    # SSL Configuration
    sslmode: str = 'require'
    
    # Performance Settings
    connection_timeout: int = 30
    command_timeout: int = 60
    
    # Storage Monitoring
    storage_limit_mb: int = 500  # Supabase free tier limit
    warning_threshold_pct: float = 80.0  # Warn at 80% usage

class SupabaseManager:
    """
    Manager class for Supabase PostgreSQL database operations.
    
    Provides connection management, health checks, and storage monitoring
    for the multi-profile trading simulator database.
    """
    
    def __init__(self, config: Optional[SupabaseConfig] = None):
        """
        Initialize the Supabase manager.
        
        Args:
            config: Optional SupabaseConfig object. If None, loads from environment.
        """
        self.config = config or self._load_config_from_env()
        self._connection_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None
        self._setup_connection_pool()
        
    def _load_config_from_env(self) -> SupabaseConfig:
        """
        Load Supabase configuration from environment variables.
        
        Environment Variables:
        - SUPABASE_URL: Full database URL (required)
        - SUPABASE_SERVICE_KEY: Service role key (required for admin operations)
        - DATABASE_URL: Alternative to SUPABASE_URL
        - ENVIRONMENT: Current environment (development/staging/production)
        
        Returns:
            SupabaseConfig object populated from environment variables.
            
        Raises:
            ValueError: If required environment variables are missing.
        """
        # Try SUPABASE_URL first, then DATABASE_URL
        database_url = os.getenv('SUPABASE_URL') or os.getenv('DATABASE_URL')
        
        if not database_url:
            raise ValueError(
                "SUPABASE_URL or DATABASE_URL environment variable is required. "
                "Example: postgresql://postgres:password@host:5432/database"
            )
        
        # Parse the database URL
        parsed_url = urlparse(database_url)
        
        if not all([parsed_url.hostname, parsed_url.username, parsed_url.password]):
            raise ValueError(
                f"Invalid database URL format. Expected: "
                f"postgresql://username:password@host:port/database"
            )
        
        # Environment-specific settings
        environment = os.getenv('ENVIRONMENT', 'development').lower()
        
        # Adjust connection pool size based on environment
        if environment == 'production':
            min_conn, max_conn = 2, 20
        elif environment == 'staging':
            min_conn, max_conn = 1, 10
        else:  # development
            min_conn, max_conn = 1, 5
        
        return SupabaseConfig(
            url=database_url,
            host=parsed_url.hostname,
            port=parsed_url.port or 5432,
            database=parsed_url.path.lstrip('/') or 'postgres',
            username=parsed_url.username,
            password=parsed_url.password,
            min_connections=min_conn,
            max_connections=max_conn,
            sslmode=os.getenv('SUPABASE_SSL_MODE', 'require'),
            connection_timeout=int(os.getenv('DB_CONNECTION_TIMEOUT', '30')),
            command_timeout=int(os.getenv('DB_COMMAND_TIMEOUT', '60'))
        )
    
    def _setup_connection_pool(self) -> None:
        """
        Set up the PostgreSQL connection pool.
        
        Creates a connection pool with the configured min/max connections
        for efficient database connection management.
        """
        try:
            self._connection_pool = psycopg2.pool.SimpleConnectionPool(
                self.config.min_connections,
                self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                sslmode=self.config.sslmode,
                connect_timeout=self.config.connection_timeout
            )
            logger.info(f"Supabase connection pool created: {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to create Supabase connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a database connection from the pool.
        
        Usage:
            with supabase_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM user_profiles")
                results = cursor.fetchall()
                
        Yields:
            psycopg2.connection: Database connection from the pool.
        """
        connection = None
        try:
            if not self._connection_pool:
                raise RuntimeError("Connection pool is not initialized")
                
            connection = self._connection_pool.getconn()
            if connection:
                yield connection
            else:
                raise RuntimeError("Failed to get connection from pool")
                
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and self._connection_pool:
                self._connection_pool.putconn(connection)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check on the Supabase database.
        
        Returns:
            Dict containing health check results with the following keys:
            - status: 'healthy', 'warning', or 'error'
            - database_accessible: bool
            - connection_pool_status: str
            - storage_usage: Dict with usage statistics
            - response_time_ms: float
            - details: List of any issues found
        """
        import time
        
        result = {
            'status': 'error',
            'database_accessible': False,
            'connection_pool_status': 'unknown',
            'storage_usage': {},
            'response_time_ms': 0,
            'details': []
        }
        
        start_time = time.time()
        
        try:
            # Test basic connectivity
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                cursor.fetchone()
                
                result['database_accessible'] = True
                
                # Check storage usage
                storage_info = self._get_storage_usage(cursor)
                result['storage_usage'] = storage_info
                
                # Check connection pool status
                if self._connection_pool:
                    available = self.config.max_connections - len([
                        c for c in self._connection_pool._pool 
                        if c and not c.closed
                    ])
                    result['connection_pool_status'] = f"{available}/{self.config.max_connections} available"
                
                # Determine overall status
                if storage_info.get('usage_pct', 0) > self.config.warning_threshold_pct:
                    result['status'] = 'warning'
                    result['details'].append(
                        f"Storage usage above {self.config.warning_threshold_pct}%"
                    )
                else:
                    result['status'] = 'healthy'
        
        except Exception as e:
            result['details'].append(f"Health check failed: {str(e)}")
            logger.error(f"Supabase health check failed: {e}")
        
        finally:
            result['response_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        return result
    
    def _get_storage_usage(self, cursor) -> Dict[str, Any]:
        """
        Get current storage usage statistics.
        
        Args:
            cursor: Database cursor for executing queries.
            
        Returns:
            Dict containing storage usage information.
        """
        try:
            # Query to get database size
            cursor.execute("""
                SELECT 
                    pg_size_pretty(pg_database_size(current_database())) as size_formatted,
                    pg_database_size(current_database()) as size_bytes,
                    current_database() as database_name
            """)
            
            result = cursor.fetchone()
            if result:
                size_bytes = result[1]
                size_mb = round(size_bytes / (1024 * 1024), 2)
                usage_pct = round((size_mb / self.config.storage_limit_mb) * 100, 2)
                
                return {
                    'database_name': result[2],
                    'size_formatted': result[0],
                    'size_bytes': size_bytes,
                    'size_mb': size_mb,
                    'limit_mb': self.config.storage_limit_mb,
                    'usage_pct': usage_pct,
                    'remaining_mb': round(self.config.storage_limit_mb - size_mb, 2)
                }
        
        except Exception as e:
            logger.warning(f"Failed to get storage usage: {e}")
            
        return {
            'error': 'Unable to determine storage usage'
        }
    
    def get_table_sizes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get size information for all tables in the trading simulator schema.
        
        Returns:
            Dict mapping table names to their size information.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size_formatted,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
                        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size_formatted,
                        pg_relation_size(schemaname||'.'||tablename) as table_size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """)
                
                tables = {}
                for row in cursor.fetchall():
                    schema, table, total_size_fmt, total_size_bytes, table_size_fmt, table_size_bytes = row
                    
                    tables[table] = {
                        'schema': schema,
                        'total_size_formatted': total_size_fmt,
                        'total_size_bytes': total_size_bytes,
                        'total_size_mb': round(total_size_bytes / (1024 * 1024), 2),
                        'table_size_formatted': table_size_fmt,
                        'table_size_bytes': table_size_bytes,
                        'table_size_mb': round(table_size_bytes / (1024 * 1024), 2),
                        'index_size_mb': round((total_size_bytes - table_size_bytes) / (1024 * 1024), 2)
                    }
                
                return tables
                
        except Exception as e:
            logger.error(f"Failed to get table sizes: {e}")
            return {}
    
    def optimize_storage(self) -> Dict[str, Any]:
        """
        Perform storage optimization operations.
        
        This includes running VACUUM, ANALYZE, and the cleanup function
        for old market data to stay within storage limits.
        
        Returns:
            Dict containing optimization results.
        """
        results = {
            'vacuum_completed': False,
            'analyze_completed': False,
            'cleanup_completed': False,
            'storage_before_mb': 0,
            'storage_after_mb': 0,
            'space_freed_mb': 0,
            'details': []
        }
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get storage before optimization
                storage_before = self._get_storage_usage(cursor)
                results['storage_before_mb'] = storage_before.get('size_mb', 0)
                
                # Run VACUUM on main tables
                main_tables = ['user_profiles', 'portfolios', 'trades', 'market_data', 
                              'performance_metrics', 'notification_settings']
                
                for table in main_tables:
                    try:
                        cursor.execute(f'VACUUM ANALYZE {table}')
                        results['details'].append(f'VACUUM ANALYZE completed for {table}')
                    except Exception as e:
                        results['details'].append(f'VACUUM ANALYZE failed for {table}: {e}')
                
                results['vacuum_completed'] = True
                results['analyze_completed'] = True
                
                # Run cleanup function for old market data
                try:
                    cursor.execute('SELECT cleanup_old_market_data()')
                    results['cleanup_completed'] = True
                    results['details'].append('Market data cleanup completed')
                except Exception as e:
                    results['details'].append(f'Market data cleanup failed: {e}')
                
                # Get storage after optimization
                storage_after = self._get_storage_usage(cursor)
                results['storage_after_mb'] = storage_after.get('size_mb', 0)
                results['space_freed_mb'] = round(
                    results['storage_before_mb'] - results['storage_after_mb'], 2
                )
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")
            results['details'].append(f'Optimization error: {e}')
        
        return results
    
    def close(self) -> None:
        """
        Close all connections and cleanup the connection pool.
        """
        if self._connection_pool:
            try:
                self._connection_pool.closeall()
                logger.info("Supabase connection pool closed")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
            finally:
                self._connection_pool = None

# Global instance for easy access
_supabase_manager: Optional[SupabaseManager] = None

def get_supabase_manager() -> SupabaseManager:
    """
    Get the global SupabaseManager instance (singleton pattern).
    
    Returns:
        SupabaseManager: Configured database manager instance.
    """
    global _supabase_manager
    
    if _supabase_manager is None:
        _supabase_manager = SupabaseManager()
    
    return _supabase_manager

def close_supabase_manager() -> None:
    """
    Close the global SupabaseManager instance and cleanup resources.
    """
    global _supabase_manager
    
    if _supabase_manager:
        _supabase_manager.close()
        _supabase_manager = None

# Utility functions for common operations
def execute_query(query: str, params: Optional[tuple] = None, fetch_results: bool = True) -> Any:
    """
    Execute a SQL query using the global Supabase manager.
    
    Args:
        query: SQL query string
        params: Optional query parameters
        fetch_results: Whether to fetch and return results
        
    Returns:
        Query results if fetch_results is True, otherwise None.
    """
    manager = get_supabase_manager()
    
    with manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        if fetch_results:
            return cursor.fetchall()
        else:
            conn.commit()
            return None

def get_storage_status() -> Dict[str, Any]:
    """
    Get current storage status for monitoring.
    
    Returns:
        Dict containing storage usage and status information.
    """
    manager = get_supabase_manager()
    health = manager.health_check()
    
    return {
        'status': health['status'],
        'storage': health.get('storage_usage', {}),
        'table_sizes': manager.get_table_sizes()
    }

# Example usage and testing
if __name__ == "__main__":
    import json
    
    # Test the Supabase configuration
    try:
        manager = get_supabase_manager()
        
        print("🔍 Supabase Health Check")
        print("=" * 50)
        
        health = manager.health_check()
        print(f"Status: {health['status']}")
        print(f"Database accessible: {health['database_accessible']}")
        print(f"Response time: {health['response_time_ms']}ms")
        
        if health.get('storage_usage'):
            storage = health['storage_usage']
            print(f"Storage: {storage.get('size_mb', 0)}MB / {storage.get('limit_mb', 500)}MB ({storage.get('usage_pct', 0)}%)")
        
        print("\n🗄️ Table Sizes")
        print("=" * 50)
        
        tables = manager.get_table_sizes()
        for table_name, info in list(tables.items())[:5]:  # Show top 5 tables
            print(f"{table_name}: {info['total_size_formatted']} (Table: {info['table_size_formatted']}, Indexes: {info['index_size_mb']}MB)")
        
        print(f"\n✅ Supabase configuration successful!")
        
    except Exception as e:
        print(f"❌ Supabase configuration failed: {e}")
        
    finally:
        close_supabase_manager()