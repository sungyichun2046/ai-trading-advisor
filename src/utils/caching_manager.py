"""
Advanced caching manager with Redis integration and memory fallback.
"""

import logging
import json
import pickle
import time
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using memory-only caching")


class CacheManager:
    """Multi-level cache manager with Redis L2 and memory L1 cache."""
    
    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600, max_memory_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_memory_size = max_memory_size
        
        # L1 Cache - Memory (fast access)
        self.memory_cache = {}
        self.memory_timestamps = {}
        self.memory_ttls = {}
        self.memory_lock = threading.RLock()
        
        # L2 Cache - Redis (persistent, cross-process)
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using memory-only cache")
                self.redis_client = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache (L1 first, then L2).
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        try:
            # Try L1 cache first
            value = self._get_from_memory(key)
            if value is not None:
                return value
            
            # Try L2 cache (Redis)
            if self.redis_client:
                value = self._get_from_redis(key)
                if value is not None:
                    # Store in L1 for faster future access
                    self._set_in_memory(key, value, self.default_ttl)
                    return value
            
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in both L1 and L2 cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            # Set in L1 cache
            self._set_in_memory(key, value, ttl)
            
            # Set in L2 cache (Redis)
            if self.redis_client:
                self._set_in_redis(key, value, ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from both cache levels."""
        try:
            # Delete from L1
            with self.memory_lock:
                self.memory_cache.pop(key, None)
                self.memory_timestamps.pop(key, None)
                self.memory_ttls.pop(key, None)
            
            # Delete from L2
            if self.redis_client:
                self.redis_client.delete(key)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def cache_cross_dag_data(self, dag_id: str, task_id: str, data: Any, ttl: Optional[int] = None) -> bool:
        """
        Cache data for cross-DAG sharing.
        
        Args:
            dag_id: DAG identifier
            task_id: Task identifier  
            data: Data to cache
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        key = f"dag_data:{dag_id}:{task_id}"
        return self.set(key, data, ttl)
    
    def get_cached_analysis_result(self, dag_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis result from DAG task.
        
        Args:
            dag_id: DAG identifier
            task_id: Task identifier
            
        Returns:
            Cached analysis result or None
        """
        key = f"dag_data:{dag_id}:{task_id}"
        result = self.get(key)
        
        if result is None:
            # Try with common analysis task patterns
            common_keys = [
                f"dag_data:{dag_id}:analyze_{task_id}",
                f"dag_data:{dag_id}:{task_id}_analysis",
                f"dag_data:analysis:{task_id}"
            ]
            
            for alt_key in common_keys:
                result = self.get(alt_key)
                if result is not None:
                    break
        
        return result
    
    def cache_dashboard_data(self, dashboard_section: str, data: Any, ttl: int = 300) -> bool:
        """Cache dashboard data with shorter TTL for real-time updates."""
        key = f"dashboard:{dashboard_section}"
        return self.set(key, data, ttl)
    
    def get_dashboard_data(self, dashboard_section: str) -> Optional[Any]:
        """Get cached dashboard data."""
        key = f"dashboard:{dashboard_section}"
        return self.get(key)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern (Redis only)."""
        try:
            if not self.redis_client:
                return 0
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Cache invalidation error for pattern {pattern}: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and performance metrics."""
        try:
            stats = {
                'memory_cache_size': len(self.memory_cache),
                'memory_max_size': self.max_memory_size,
                'redis_available': self.redis_client is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.redis_client:
                try:
                    redis_info = self.redis_client.info()
                    stats.update({
                        'redis_memory_used': redis_info.get('used_memory_human', 'N/A'),
                        'redis_connected_clients': redis_info.get('connected_clients', 0),
                        'redis_keyspace_hits': redis_info.get('keyspace_hits', 0),
                        'redis_keyspace_misses': redis_info.get('keyspace_misses', 0)
                    })
                    
                    # Calculate hit rate
                    hits = redis_info.get('keyspace_hits', 0)
                    misses = redis_info.get('keyspace_misses', 0)
                    if hits + misses > 0:
                        stats['redis_hit_rate'] = hits / (hits + misses)
                    else:
                        stats['redis_hit_rate'] = 0.0
                        
                except Exception as e:
                    logger.warning(f"Could not get Redis stats: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def _get_from_memory(self, key: str) -> Any:
        """Get value from L1 memory cache."""
        with self.memory_lock:
            if key not in self.memory_cache:
                return None
            
            # Check TTL
            if key in self.memory_ttls:
                expiry_time = self.memory_timestamps.get(key, 0) + self.memory_ttls[key]
                if time.time() > expiry_time:
                    # Expired, remove from cache
                    self.memory_cache.pop(key, None)
                    self.memory_timestamps.pop(key, None)
                    self.memory_ttls.pop(key, None)
                    return None
            
            return self.memory_cache[key]
    
    def _set_in_memory(self, key: str, value: Any, ttl: int) -> None:
        """Set value in L1 memory cache."""
        with self.memory_lock:
            # Enforce max memory size by removing oldest entries
            while len(self.memory_cache) >= self.max_memory_size:
                if not self.memory_timestamps:
                    break
                oldest_key = min(self.memory_timestamps.keys(), key=lambda k: self.memory_timestamps[k])
                self.memory_cache.pop(oldest_key, None)
                self.memory_timestamps.pop(oldest_key, None) 
                self.memory_ttls.pop(oldest_key, None)
            
            self.memory_cache[key] = value
            self.memory_timestamps[key] = time.time()
            self.memory_ttls[key] = ttl
    
    def _get_from_redis(self, key: str) -> Any:
        """Get value from L2 Redis cache."""
        try:
            serialized_value = self.redis_client.get(key)
            if serialized_value is None:
                return None
            
            # Try JSON first, then pickle
            try:
                return json.loads(serialized_value)
            except (json.JSONDecodeError, TypeError):
                try:
                    import base64
                    return pickle.loads(base64.b64decode(serialized_value))
                except Exception:
                    logger.warning(f"Could not deserialize value for key {key}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            return None
    
    def _set_in_redis(self, key: str, value: Any, ttl: int) -> None:
        """Set value in L2 Redis cache."""
        try:
            # Try JSON serialization first
            try:
                serialized_value = json.dumps(value)
            except (TypeError, ValueError):
                # Fallback to pickle for complex objects
                import base64
                serialized_value = base64.b64encode(pickle.dumps(value)).decode()
            
            self.redis_client.setex(key, ttl, serialized_value)
            
        except Exception as e:
            logger.warning(f"Redis set error for key {key}: {e}")


# Global cache manager instance
cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global cache_manager
    if cache_manager is None:
        try:
            from .config_loader import get_cache_config
            config = get_cache_config()
            cache_manager = CacheManager(
                redis_url=config.get('redis_url'),
                default_ttl=config.get('default_ttl', 3600),
                max_memory_size=config.get('max_memory_size', 1000)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize cache manager with config: {e}")
            cache_manager = CacheManager()
    return cache_manager

# Convenience functions
def cache_cross_dag_data(dag_id: str, task_id: str, data: Any, ttl: Optional[int] = None) -> bool:
    """Cache data for cross-DAG sharing."""
    return get_cache_manager().cache_cross_dag_data(dag_id, task_id, data, ttl)

def get_cached_analysis_result(dag_id: str, task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached analysis result."""
    return get_cache_manager().get_cached_analysis_result(dag_id, task_id)