"""Tests for caching manager utilities."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from src.utils.caching_manager import CacheManager, get_cache_manager, cache_cross_dag_data


class TestCacheManager:
    """Test caching manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create cache manager without Redis for testing
        self.cache_manager = CacheManager(redis_url=None, default_ttl=300, max_memory_size=100)
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        assert self.cache_manager.default_ttl == 300
        assert self.cache_manager.max_memory_size == 100
        assert self.cache_manager.redis_client is None
        assert len(self.cache_manager.memory_cache) == 0
    
    def test_set_and_get_memory_cache(self):
        """Test setting and getting values in memory cache."""
        # Set value
        success = self.cache_manager.set('test_key', 'test_value', 100)
        assert success is True
        
        # Get value
        result = self.cache_manager.get('test_key')
        assert result == 'test_value'
    
    def test_get_nonexistent_key(self):
        """Test getting non-existent key returns default."""
        result = self.cache_manager.get('nonexistent', 'default_value')
        assert result == 'default_value'
    
    def test_cache_expiry(self):
        """Test cache expiry functionality."""
        # Set with very short TTL
        self.cache_manager.set('expiry_test', 'value', 1)
        
        # Should exist immediately
        result = self.cache_manager.get('expiry_test')
        assert result == 'value'
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Should be expired now
        result = self.cache_manager.get('expiry_test', 'expired')
        assert result == 'expired'
    
    def test_cache_delete(self):
        """Test cache deletion."""
        # Set value
        self.cache_manager.set('delete_test', 'value')
        assert self.cache_manager.get('delete_test') == 'value'
        
        # Delete value
        success = self.cache_manager.delete('delete_test')
        assert success is True
        
        # Should be gone
        result = self.cache_manager.get('delete_test', 'not_found')
        assert result == 'not_found'
    
    def test_memory_cache_size_limit(self):
        """Test memory cache size limitation."""
        # Fill cache beyond limit
        for i in range(150):  # More than max_memory_size of 100
            self.cache_manager.set(f'key_{i}', f'value_{i}')
        
        # Should have trimmed to max size
        assert len(self.cache_manager.memory_cache) <= 100
        
        # Newer entries should still exist
        assert self.cache_manager.get('key_149') == 'value_149'
    
    def test_cache_cross_dag_data(self):
        """Test cross-DAG data caching."""
        test_data = {'analysis': 'result', 'confidence': 0.8}
        
        success = self.cache_manager.cache_cross_dag_data('analysis', 'technical_indicators', test_data)
        assert success is True
        
        # Should be retrievable
        result = self.cache_manager.get_cached_analysis_result('analysis', 'technical_indicators')
        assert result == test_data
    
    def test_get_cached_analysis_result_fallback_keys(self):
        """Test cached analysis result with fallback key patterns."""
        test_data = {'signals': ['buy', 'hold']}
        
        # Store with alternative key pattern
        self.cache_manager.set('dag_data:analysis:analyze_sentiment', test_data)
        
        # Should find with original task_id
        result = self.cache_manager.get_cached_analysis_result('analysis', 'sentiment')
        assert result == test_data
    
    def test_cache_dashboard_data(self):
        """Test dashboard data caching."""
        dashboard_data = {'portfolio_value': 100000, 'return_pct': 5.2}
        
        success = self.cache_manager.cache_dashboard_data('portfolio_summary', dashboard_data, 60)
        assert success is True
        
        # Should be retrievable
        result = self.cache_manager.get_dashboard_data('portfolio_summary')
        assert result == dashboard_data
    
    def test_get_cache_stats(self):
        """Test cache statistics retrieval."""
        # Add some data
        self.cache_manager.set('stats_test', 'value')
        
        stats = self.cache_manager.get_cache_stats()
        
        assert 'memory_cache_size' in stats
        assert 'redis_available' in stats
        assert 'timestamp' in stats
        assert stats['memory_cache_size'] >= 1
        assert stats['redis_available'] is False
    
    def test_cache_with_different_data_types(self):
        """Test caching with different data types."""
        test_cases = [
            ('string', 'test_string'),
            ('integer', 42),
            ('float', 3.14),
            ('list', [1, 2, 3]),
            ('dict', {'key': 'value'}),
            ('boolean', True),
            ('none', None)
        ]
        
        for key, value in test_cases:
            self.cache_manager.set(key, value)
            result = self.cache_manager.get(key)
            assert result == value
    
    def test_error_handling_in_get(self):
        """Test error handling in cache get operations."""
        # Simulate error in memory access
        with patch.object(self.cache_manager, '_get_from_memory', side_effect=Exception("Memory error")):
            result = self.cache_manager.get('error_key', 'default')
            assert result == 'default'
    
    def test_error_handling_in_set(self):
        """Test error handling in cache set operations."""
        # Simulate error in memory set
        with patch.object(self.cache_manager, '_set_in_memory', side_effect=Exception("Set error")):
            result = self.cache_manager.set('error_key', 'value')
            assert result is False


class TestCacheManagerWithRedis:
    """Test cache manager with Redis integration."""
    
    def setup_method(self):
        """Set up test fixtures with mocked Redis."""
        self.mock_redis = MagicMock()
        self.mock_redis.ping.return_value = True
        
    def test_cache_manager_with_redis_connection(self):
        """Test cache manager initialization with Redis."""
        with patch('src.utils.caching_manager.redis.from_url', return_value=self.mock_redis):
            cache_manager = CacheManager(redis_url='redis://localhost:6379/0')
            assert cache_manager.redis_client is not None
    
    def test_cache_manager_redis_connection_failure(self):
        """Test cache manager handling Redis connection failure."""
        mock_redis_fail = MagicMock()
        mock_redis_fail.ping.side_effect = Exception("Connection failed")
        
        with patch('src.utils.caching_manager.redis.from_url', return_value=mock_redis_fail):
            cache_manager = CacheManager(redis_url='redis://localhost:6379/0')
            assert cache_manager.redis_client is None
    
    def test_redis_get_and_set(self):
        """Test Redis get and set operations."""
        self.mock_redis.get.return_value = '{"test": "value"}'
        self.mock_redis.setex.return_value = True
        
        with patch('src.utils.caching_manager.redis.from_url', return_value=self.mock_redis):
            cache_manager = CacheManager(redis_url='redis://localhost:6379/0')
            
            # Test set
            success = cache_manager.set('redis_test', {'test': 'value'})
            assert success is True
            self.mock_redis.setex.assert_called()
            
            # Test get (with L1 cache miss)
            cache_manager.memory_cache.clear()  # Force L2 lookup
            result = cache_manager.get('redis_test')
            assert result == {'test': 'value'}
            self.mock_redis.get.assert_called()
    
    def test_redis_serialization_fallback(self):
        """Test Redis serialization fallback to pickle."""
        # Mock complex object that can't be JSON serialized
        complex_obj = {'function': lambda x: x}
        
        self.mock_redis.setex.return_value = True
        
        with patch('src.utils.caching_manager.redis.from_url', return_value=self.mock_redis):
            cache_manager = CacheManager(redis_url='redis://localhost:6379/0')
            
            # Should use pickle fallback
            with patch('pickle.dumps', return_value=b'pickled_data') as mock_pickle:
                cache_manager._set_in_redis('complex_key', complex_obj, 300)
                mock_pickle.assert_called_once()
    
    def test_invalidate_pattern(self):
        """Test pattern-based cache invalidation."""
        self.mock_redis.keys.return_value = ['pattern_key1', 'pattern_key2']
        self.mock_redis.delete.return_value = 2
        
        with patch('src.utils.caching_manager.redis.from_url', return_value=self.mock_redis):
            cache_manager = CacheManager(redis_url='redis://localhost:6379/0')
            
            count = cache_manager.invalidate_pattern('pattern_*')
            assert count == 2
            self.mock_redis.keys.assert_called_with('pattern_*')
            self.mock_redis.delete.assert_called_with('pattern_key1', 'pattern_key2')


class TestCacheManagerConvenienceFunctions:
    """Test convenience functions for cache manager."""
    
    def test_get_cache_manager_singleton(self):
        """Test cache manager singleton pattern."""
        # Reset global cache manager
        import src.utils.caching_manager
        src.utils.caching_manager.cache_manager = None
        
        # Should create new instance
        manager1 = get_cache_manager()
        assert manager1 is not None
        
        # Should return same instance
        manager2 = get_cache_manager()
        assert manager1 is manager2
    
    def test_cache_cross_dag_data_convenience(self):
        """Test cache_cross_dag_data convenience function."""
        with patch('src.utils.caching_manager.get_cache_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.cache_cross_dag_data.return_value = True
            mock_get_manager.return_value = mock_manager
            
            result = cache_cross_dag_data('test_dag', 'test_task', {'data': 'test'})
            assert result is True
            mock_manager.cache_cross_dag_data.assert_called_once()


class TestCachePerformance:
    """Test cache performance and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = CacheManager(redis_url=None, max_memory_size=10)
    
    def test_concurrent_access(self):
        """Test concurrent access to cache."""
        import threading
        
        def worker(thread_id):
            for i in range(10):
                key = f'thread_{thread_id}_key_{i}'
                self.cache_manager.set(key, f'value_{i}')
                result = self.cache_manager.get(key)
                assert result == f'value_{i}'
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Cache should still be functional
        self.cache_manager.set('final_test', 'final_value')
        assert self.cache_manager.get('final_test') == 'final_value'
    
    def test_large_value_caching(self):
        """Test caching large values."""
        large_value = 'x' * 10000  # 10KB string
        
        success = self.cache_manager.set('large_value', large_value)
        assert success is True
        
        result = self.cache_manager.get('large_value')
        assert result == large_value
    
    def test_cache_behavior_under_memory_pressure(self):
        """Test cache behavior when approaching memory limits."""
        # Fill cache to capacity
        for i in range(15):  # More than max_memory_size of 10
            self.cache_manager.set(f'pressure_key_{i}', f'value_{i}')
        
        # Older entries should be evicted
        assert len(self.cache_manager.memory_cache) <= 10
        
        # Newest entries should still exist
        assert self.cache_manager.get('pressure_key_14') == 'value_14'
        assert self.cache_manager.get('pressure_key_0', 'not_found') == 'not_found'