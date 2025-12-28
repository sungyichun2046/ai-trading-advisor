"""Tests for configuration loader utilities."""

import os
import tempfile
import yaml
import json
import pytest
from unittest.mock import Mock, patch, mock_open
from src.utils.config_loader import ConfigLoader, load_yaml_config, load_environment_config


class TestConfigLoader:
    """Test configuration loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_loader = ConfigLoader()
    
    def test_load_yaml_config_success(self):
        """Test successful YAML configuration loading."""
        test_config = {'test_key': 'test_value', 'nested': {'key': 'value'}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            result = self.config_loader.load_yaml_config(temp_path)
            assert result == test_config
        finally:
            os.unlink(temp_path)
    
    def test_load_yaml_config_file_not_found(self):
        """Test YAML loading with non-existent file."""
        result = self.config_loader.load_yaml_config('/non/existent/file.yaml')
        assert 'environment' in result  # Should return default config
        assert result['environment'] == 'dev'
    
    def test_load_yaml_config_with_schema_validation(self):
        """Test YAML loading with schema validation."""
        test_config = {'required_key': 'value', 'number': 42}
        schema = {'required_key': str, 'number': int}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            result = self.config_loader.load_yaml_config(temp_path, schema)
            assert result == test_config
        finally:
            os.unlink(temp_path)
    
    def test_load_yaml_config_schema_validation_failure(self):
        """Test YAML loading with schema validation failure."""
        test_config = {'required_key': 123}  # Wrong type
        schema = {'required_key': str, 'missing_key': str}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            result = self.config_loader.load_yaml_config(temp_path, schema)
            assert 'environment' in result  # Should fallback to default
        finally:
            os.unlink(temp_path)
    
    def test_load_json_config_success(self):
        """Test successful JSON configuration loading."""
        test_config = {'test_key': 'test_value', 'number': 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name
        
        try:
            result = self.config_loader.load_json_config(temp_path)
            assert result == test_config
        finally:
            os.unlink(temp_path)
    
    def test_load_json_config_file_not_found(self):
        """Test JSON loading with non-existent file."""
        result = self.config_loader.load_json_config('/non/existent/file.json')
        assert result == {}
    
    def test_load_environment_config_auto_detect(self):
        """Test environment configuration auto-detection."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'staging'}, clear=False):
            result = self.config_loader.load_environment_config()
            assert 'environment' in result
    
    def test_load_environment_config_specific_env(self):
        """Test loading specific environment configuration."""
        result = self.config_loader.load_environment_config('dev')
        assert 'environment' in result
    
    def test_get_database_config_with_env_vars(self):
        """Test database configuration with environment variables."""
        with patch.dict(os.environ, {
            'POSTGRES_HOST': 'test-host',
            'POSTGRES_PORT': '5433',
            'POSTGRES_DB': 'test-db'
        }, clear=False):
            result = self.config_loader.get_database_config()
            assert result['host'] == 'test-host'
            assert result['port'] == '5433'
            assert result['database'] == 'test-db'
    
    def test_get_cache_config_defaults(self):
        """Test cache configuration with defaults."""
        result = self.config_loader.get_cache_config()
        assert 'redis_url' in result
        assert 'default_ttl' in result
        assert result['default_ttl'] == 3600
    
    def test_detect_environment_from_env_vars(self):
        """Test environment detection from various environment variables."""
        # Test production detection
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}, clear=False):
            result = self.config_loader._detect_environment()
            assert result == 'prod'
        
        # Test staging detection
        with patch.dict(os.environ, {'ENV': 'staging'}, clear=False):
            result = self.config_loader._detect_environment()
            assert result == 'staging'
        
        # Test development detection
        with patch.dict(os.environ, {'ENVIRONMENT': 'dev'}, clear=False):
            result = self.config_loader._detect_environment()
            assert result == 'dev'
    
    def test_detect_environment_from_platform_indicators(self):
        """Test environment detection from platform indicators."""
        # Test production detection via platform indicators
        with patch.dict(os.environ, {'KUBERNETES_SERVICE_HOST': 'k8s'}, clear=False):
            result = self.config_loader._detect_environment()
            assert result == 'prod'
        
        # Test development detection via virtual env
        with patch.dict(os.environ, {'VIRTUAL_ENV': '/venv'}, clear=False):
            result = self.config_loader._detect_environment()
            assert result == 'dev'
    
    def test_config_hot_reloading(self):
        """Test configuration hot-reloading on file changes."""
        test_config = {'version': 1}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            # Load first time
            result1 = self.config_loader.load_yaml_config(temp_path)
            assert result1['version'] == 1
            
            # Update file
            import time
            time.sleep(1)  # Ensure different timestamp
            updated_config = {'version': 2}
            with open(temp_path, 'w') as f:
                yaml.dump(updated_config, f)
            
            # Load again (should reload)
            result2 = self.config_loader.load_yaml_config(temp_path)
            assert result2['version'] == 2
            
        finally:
            os.unlink(temp_path)


class TestConfigLoaderConvenienceFunctions:
    """Test convenience functions for configuration loading."""
    
    def test_load_yaml_config_function(self):
        """Test load_yaml_config convenience function."""
        with patch('src.utils.config_loader.config_loader.load_yaml_config') as mock_load:
            mock_load.return_value = {'test': 'value'}
            result = load_yaml_config('test.yaml')
            assert result == {'test': 'value'}
            mock_load.assert_called_once_with('test.yaml', None)
    
    def test_load_environment_config_function(self):
        """Test load_environment_config convenience function."""
        with patch('src.utils.config_loader.config_loader.load_environment_config') as mock_load:
            mock_load.return_value = {'env': 'test'}
            result = load_environment_config('test')
            assert result == {'env': 'test'}
            mock_load.assert_called_once_with('test')


class TestConfigLoaderErrorHandling:
    """Test error handling in configuration loading."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_loader = ConfigLoader()
    
    def test_load_yaml_config_invalid_yaml(self):
        """Test YAML loading with invalid YAML content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            temp_path = f.name
        
        try:
            result = self.config_loader.load_yaml_config(temp_path)
            assert 'environment' in result  # Should return default config
        finally:
            os.unlink(temp_path)
    
    def test_load_json_config_invalid_json(self):
        """Test JSON loading with invalid JSON content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json content}')
            temp_path = f.name
        
        try:
            result = self.config_loader.load_json_config(temp_path)
            assert result == {}  # Should return empty dict on error
        finally:
            os.unlink(temp_path)
    
    def test_get_database_config_error_handling(self):
        """Test database configuration error handling."""
        with patch.object(self.config_loader, 'load_environment_config', side_effect=Exception("Test error")):
            result = self.config_loader.get_database_config()
            assert 'host' in result  # Should return default config
            assert result['host'] == 'localhost'
    
    def test_get_cache_config_error_handling(self):
        """Test cache configuration error handling."""
        with patch.object(self.config_loader, 'load_environment_config', side_effect=Exception("Test error")):
            result = self.config_loader.get_cache_config()
            assert 'redis_url' in result  # Should return default config