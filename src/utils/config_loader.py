"""
Configuration loading and validation utilities with hot-reloading support.
"""

import logging
import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Advanced configuration loader with validation and hot-reloading."""
    
    def __init__(self):
        self.loaded_configs = {}
        self.config_timestamps = {}
    
    def load_yaml_config(self, file_path: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load YAML configuration file with validation and caching.
        
        Args:
            file_path: Path to YAML configuration file
            schema: Optional schema for validation
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            full_path = Path(file_path).resolve()
            
            # Check if file exists
            if not full_path.exists():
                logger.warning(f"Config file not found: {file_path}, using default config")
                return self._get_default_config(file_path)
            
            # Check if we need to reload based on modification time
            current_mtime = full_path.stat().st_mtime
            cached_mtime = self.config_timestamps.get(str(full_path))
            
            if str(full_path) in self.loaded_configs and current_mtime <= (cached_mtime or 0):
                return self.loaded_configs[str(full_path)]
            
            # Load and parse YAML
            with open(full_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file) or {}
            
            # Validate schema if provided
            if schema:
                validation_errors = self._validate_config_schema(config, schema)
                if validation_errors:
                    logger.warning(f"Config validation errors: {validation_errors}")
                    return self._get_default_config(file_path)
            
            # Cache the loaded config
            self.loaded_configs[str(full_path)] = config
            self.config_timestamps[str(full_path)] = current_mtime
            
            logger.info(f"Successfully loaded config from {file_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return self._get_default_config(file_path)
    
    def load_json_config(self, file_path: str) -> Dict[str, Any]:
        """Load JSON configuration file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Failed to load JSON config from {file_path}: {e}")
            return {}
    
    def load_environment_config(self, env: Optional[str] = None) -> Dict[str, Any]:
        """
        Load environment-specific configuration.
        
        Args:
            env: Environment name (dev/staging/prod). If None, auto-detect.
            
        Returns:
            Environment-specific configuration
        """
        try:
            if env is None:
                env = self._detect_environment()
            
            config_file = f"config/environments/{env}.yaml"
            
            # Try to load environment-specific config
            if os.path.exists(config_file):
                return self.load_yaml_config(config_file)
            
            # Fallback to default config with environment overrides
            base_config = self.load_yaml_config("config/default.yaml")
            env_overrides = os.environ.get(f'{env.upper()}_CONFIG_OVERRIDES')
            
            if env_overrides:
                try:
                    overrides = json.loads(env_overrides)
                    base_config.update(overrides)
                except Exception as e:
                    logger.warning(f"Failed to parse environment overrides: {e}")
            
            return base_config
            
        except Exception as e:
            logger.error(f"Failed to load environment config for {env}: {e}")
            return self._get_default_environment_config()
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration with environment-specific settings."""
        try:
            config = self.load_environment_config()
            db_config = config.get('database', {})
            
            # Override with environment variables if available
            env_vars = {
                'host': 'POSTGRES_HOST',
                'port': 'POSTGRES_PORT', 
                'database': 'POSTGRES_DB',
                'username': 'POSTGRES_USER',
                'password': 'POSTGRES_PASSWORD'
            }
            
            for key, env_var in env_vars.items():
                if env_var in os.environ:
                    db_config[key] = os.environ[env_var]
            
            return db_config
            
        except Exception as e:
            logger.error(f"Failed to get database config: {e}")
            return self._get_default_database_config()
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get caching configuration."""
        try:
            config = self.load_environment_config()
            cache_config = config.get('cache', {})
            
            # Set defaults
            cache_config.setdefault('redis_url', os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
            cache_config.setdefault('default_ttl', 3600)
            cache_config.setdefault('max_memory_size', '100MB')
            
            return cache_config
            
        except Exception as e:
            logger.error(f"Failed to get cache config: {e}")
            return {'redis_url': 'redis://localhost:6379/0', 'default_ttl': 3600, 'max_memory_size': '100MB'}
    
    def _validate_config_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> list:
        """Basic schema validation for configuration."""
        errors = []
        
        for key, expected_type in schema.items():
            if key not in config:
                errors.append(f"Missing required key: {key}")
            elif not isinstance(config[key], expected_type):
                errors.append(f"Invalid type for {key}: expected {expected_type}, got {type(config[key])}")
        
        return errors
    
    def _detect_environment(self) -> str:
        """Auto-detect current environment."""
        # First check explicit environment variables
        env = os.environ.get('ENVIRONMENT', os.environ.get('ENV', '')).lower()
        
        if env in ['prod', 'production']:
            return 'prod'
        elif env in ['staging', 'stage']:
            return 'staging'
        elif env in ['dev', 'development']:
            return 'dev'
        
        # Check for production platform indicators
        prod_indicators = [
            'KUBERNETES_SERVICE_HOST',  # Running in k8s
            'AWS_EXECUTION_ENV',        # Running in AWS
            'HEROKU_APP_NAME',          # Running on Heroku
        ]
        
        if any(os.environ.get(indicator) for indicator in prod_indicators):
            return 'prod'
        
        # Check for development indicators
        dev_indicators = [
            'VIRTUAL_ENV',              # Virtual environment
            'CONDA_DEFAULT_ENV',        # Conda environment
        ]
        
        if any(os.environ.get(indicator) for indicator in dev_indicators):
            return 'dev'
            
        return 'dev'  # Default
    
    def _get_default_config(self, file_path: str) -> Dict[str, Any]:
        """Get default configuration based on file type."""
        if 'dependencies' in file_path:
            return {
                'workflow': {
                    'execution_order': ['data_collection', 'analysis', 'trading'],
                    'parallel_analysis': True,
                    'enable_consensus': True
                },
                'timeouts': {'default': 300, 'long_running': 1800}
            }
        elif 'database' in file_path:
            return self._get_default_database_config()
        else:
            return {'environment': 'dev', 'debug': True}
    
    def _get_default_database_config(self) -> Dict[str, Any]:
        """Default database configuration."""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'airflow',
            'username': 'airflow', 
            'password': 'airflow',
            'pool_size': 10,
            'max_overflow': 20
        }
    
    def _get_default_environment_config(self) -> Dict[str, Any]:
        """Default environment configuration."""
        return {
            'environment': 'dev',
            'debug': True,
            'database': self._get_default_database_config(),
            'cache': {'redis_url': 'redis://localhost:6379/0', 'default_ttl': 3600}
        }


# Global config loader instance
config_loader = ConfigLoader()

# Convenience functions
def load_yaml_config(file_path: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML configuration file."""
    return config_loader.load_yaml_config(file_path, schema)

def load_environment_config(env: Optional[str] = None) -> Dict[str, Any]:
    """Load environment-specific configuration."""
    return config_loader.load_environment_config(env)

def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    return config_loader.get_database_config()

def get_cache_config() -> Dict[str, Any]:
    """Get cache configuration."""
    return config_loader.get_cache_config()