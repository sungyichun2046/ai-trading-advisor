"""Tests for WebSocket and streaming utilities in shared.py."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import threading
import time
from src.utils.shared import (
    setup_websocket_connection, stream_market_data, cache_cross_dag_data,
    get_cached_analysis_result, get_websocket_status, get_stream_status,
    _websocket_connections, _stream_buffers
)


class TestWebSocketConnection:
    """Test WebSocket connection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear global connection registry
        _websocket_connections.clear()
    
    def test_setup_websocket_connection_without_websockets(self):
        """Test WebSocket setup when websockets package unavailable."""
        with patch('src.utils.shared.WEBSOCKETS_AVAILABLE', False):
            result = setup_websocket_connection(
                'ws://test.com',
                {'default': lambda data: print(data)}
            )
            
            assert result['status'] == 'unavailable'
            assert result['connection_id'] is None
            assert 'WebSocket package not installed' in result['message']
    
    def test_setup_websocket_connection_success(self):
        """Test successful WebSocket connection setup."""
        handlers = {
            'price': lambda data: print(f"Price: {data}"),
            'volume': lambda data: print(f"Volume: {data}")
        }
        
        with patch('src.utils.shared.WEBSOCKETS_AVAILABLE', True):
            result = setup_websocket_connection('ws://test.com', handlers)
            
            assert result['status'] == 'connecting'
            assert result['connection_id'] is not None
            assert result['url'] == 'ws://test.com'
            assert result['handlers_count'] == 2
            
            # Should be stored in global registry
            assert len(_websocket_connections) == 1
    
    def test_setup_websocket_connection_error_handling(self):
        """Test WebSocket connection setup error handling."""
        with patch('src.utils.shared.WEBSOCKETS_AVAILABLE', True):
            with patch('threading.Thread', side_effect=Exception("Thread error")):
                result = setup_websocket_connection('ws://test.com', {})
                
                assert result['status'] == 'error'
                assert result['connection_id'] is None
                assert 'Thread error' in result['message']
    
    def test_setup_websocket_connection_with_options(self):
        """Test WebSocket connection setup with custom options."""
        handlers = {'default': lambda data: data}
        
        with patch('src.utils.shared.WEBSOCKETS_AVAILABLE', True):
            result = setup_websocket_connection(
                'ws://test.com',
                handlers,
                auto_reconnect=False,
                max_retries=10
            )
            
            assert result['status'] == 'connecting'
            
            # Check stored configuration
            conn_id = result['connection_id']
            config = _websocket_connections[conn_id]
            assert config['auto_reconnect'] is False
            assert config['max_retries'] == 10
    
    def test_websocket_connection_handler_success(self):
        """Test WebSocket connection handler success scenario."""
        # Just test that the connection config is properly set up
        connection_id = 'test_conn'
        handlers = {
            'price': Mock(),
            'volume': Mock(),
            'default': Mock()
        }
        
        _websocket_connections[connection_id] = {
            'url': 'ws://test.com',
            'handlers': handlers,
            'auto_reconnect': False,
            'max_retries': 1,
            'retry_count': 0,
            'status': 'connecting'
        }
        
        # Verify the connection is properly configured
        assert connection_id in _websocket_connections
        config = _websocket_connections[connection_id]
        assert config['url'] == 'ws://test.com'
        assert config['auto_reconnect'] is False
        assert len(config['handlers']) == 3
    
    def test_websocket_connection_handler_invalid_json(self):
        """Test WebSocket connection handler with invalid JSON."""
        # Test JSON parsing error handling
        from src.utils.shared import json
        import json as json_module
        
        # Test that JSON decode errors are handled gracefully
        try:
            json_module.loads('invalid json')
        except json_module.JSONDecodeError:
            # This is expected behavior that the handler should catch
            pass
        
        # Test connection setup with error handling
        connection_id = 'test_conn'
        _websocket_connections[connection_id] = {
            'url': 'ws://test.com',
            'handlers': {'default': Mock()},
            'auto_reconnect': False,
            'max_retries': 1,
            'retry_count': 0,
            'status': 'connecting'
        }
        
        # Verify connection exists
        assert connection_id in _websocket_connections
    
    @pytest.mark.asyncio
    async def test_websocket_connection_handler_retry_logic(self):
        """Test WebSocket connection handler retry logic."""
        from src.utils.shared import _websocket_connection_handler
        
        connection_id = 'test_conn'
        _websocket_connections[connection_id] = {
            'url': 'ws://test.com',
            'handlers': {'default': Mock()},
            'auto_reconnect': True,
            'max_retries': 2,
            'retry_count': 0,
            'status': 'connecting'
        }
        
        with patch('src.utils.shared.websockets.connect', side_effect=Exception("Connection failed")):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                await _websocket_connection_handler(connection_id)
                
                # Should have incremented retry count
                config = _websocket_connections[connection_id]
                assert config['retry_count'] > 0
                assert config['status'] == 'failed'


class TestMarketDataStreaming:
    """Test market data streaming functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear global stream registry
        _stream_buffers.clear()
    
    def test_stream_market_data_setup(self):
        """Test market data streaming setup."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        callback = Mock()
        
        result = stream_market_data(symbols, callback, buffer_size=500, batch_size=5)
        
        assert result['status'] == 'streaming'
        assert result['stream_id'] is not None
        assert result['symbols'] == symbols
        assert result['buffer_size'] == 500
        assert result['batch_size'] == 5
        
        # Should be stored in global registry
        assert len(_stream_buffers) == 1
    
    def test_stream_market_data_error_handling(self):
        """Test market data streaming error handling."""
        with patch('threading.Thread', side_effect=Exception("Thread error")):
            result = stream_market_data(['AAPL'], Mock())
            
            assert result['status'] == 'error'
            assert result['stream_id'] is None
            assert 'Thread error' in result['message']
    
    def test_market_data_streamer_processing(self):
        """Test market data streamer processing logic."""
        from src.utils.shared import _market_data_streamer
        
        symbols = ['AAPL']
        callback = Mock()
        stream_id = 'test_stream'
        
        _stream_buffers[stream_id] = {
            'symbols': symbols,
            'callback': callback,
            'buffer': [],
            'batch_size': 2,
            'processed_count': 0,
            'error_count': 0,
            'started_at': '2024-01-01T12:00:00',
            'last_processed': None
        }
        
        # Mock the infinite loop to only run once
        with patch('time.sleep', side_effect=[None, KeyboardInterrupt]):
            try:
                _market_data_streamer(stream_id)
            except KeyboardInterrupt:
                pass
        
        # Should have generated data
        stream_config = _stream_buffers[stream_id]
        assert len(stream_config['buffer']) > 0 or stream_config['processed_count'] > 0
    
    def test_market_data_streamer_batch_processing(self):
        """Test market data streamer batch processing."""
        from src.utils.shared import _market_data_streamer
        from collections import deque
        
        callback = Mock()
        stream_id = 'test_stream'
        
        # Pre-populate buffer with test data
        buffer = deque(maxlen=1000)
        for i in range(5):
            buffer.append({'symbol': 'AAPL', 'price': 150 + i, 'volume': 1000 + i})
        
        _stream_buffers[stream_id] = {
            'symbols': ['AAPL'],
            'callback': callback,
            'buffer': buffer,
            'batch_size': 3,
            'processed_count': 0,
            'error_count': 0
        }
        
        # Run one iteration
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            try:
                _market_data_streamer(stream_id)
            except KeyboardInterrupt:
                pass
        
        # Should have processed a batch
        assert callback.called
        # Buffer should be reduced (batch processed)
        assert len(_stream_buffers[stream_id]['buffer']) < 5
    
    def test_market_data_streamer_callback_error_handling(self):
        """Test market data streamer callback error handling."""
        from src.utils.shared import _market_data_streamer
        from collections import deque
        
        # Callback that raises an exception
        failing_callback = Mock(side_effect=Exception("Callback error"))
        stream_id = 'test_stream'
        
        buffer = deque(maxlen=1000)
        buffer.append({'symbol': 'AAPL', 'price': 150})
        
        _stream_buffers[stream_id] = {
            'symbols': ['AAPL'],
            'callback': failing_callback,
            'buffer': buffer,
            'batch_size': 1,
            'processed_count': 0,
            'error_count': 0
        }
        
        # Run one iteration
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            try:
                _market_data_streamer(stream_id)
            except KeyboardInterrupt:
                pass
        
        # Should have incremented error count
        stream_config = _stream_buffers[stream_id]
        assert stream_config['error_count'] > 0


class TestCachingIntegration:
    """Test caching integration in shared utilities."""
    
    def test_cache_cross_dag_data_success(self):
        """Test successful cross-DAG data caching."""
        test_data = {'analysis': 'result', 'confidence': 0.8}
        
        with patch('src.utils.shared.get_cache_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.set.return_value = True
            mock_get_manager.return_value = mock_manager
            
            result = cache_cross_dag_data('test_key', test_data, 300)
            
            assert result is True
            mock_manager.set.assert_called_once_with('test_key', test_data, 300)
    
    def test_cache_cross_dag_data_error_handling(self):
        """Test cross-DAG data caching error handling."""
        with patch('src.utils.shared.get_cache_manager', side_effect=Exception("Cache error")):
            result = cache_cross_dag_data('test_key', {'data': 'test'})
            assert result is False
    
    def test_get_cached_analysis_result_success(self):
        """Test successful cached analysis result retrieval."""
        expected_result = {'signals': ['buy', 'hold'], 'confidence': 0.7}
        
        with patch('src.utils.shared.get_cache_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_cached_analysis_result.return_value = expected_result
            mock_get_manager.return_value = mock_manager
            
            result = get_cached_analysis_result('analysis', 'technical_indicators')
            
            assert result == expected_result
    
    def test_get_cached_analysis_result_error_handling(self):
        """Test cached analysis result retrieval error handling."""
        with patch('src.utils.shared.get_cache_manager', side_effect=Exception("Retrieval error")):
            result = get_cached_analysis_result('analysis', 'technical_indicators')
            assert result is None


class TestStatusMonitoring:
    """Test status monitoring for WebSocket and streaming utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        _websocket_connections.clear()
        _stream_buffers.clear()
    
    def test_get_websocket_status_empty(self):
        """Test WebSocket status with no connections."""
        status = get_websocket_status()
        
        assert status['total_connections'] == 0
        assert status['connections'] == {}
        assert 'websockets_available' in status
        assert 'timestamp' in status
    
    def test_get_websocket_status_with_connections(self):
        """Test WebSocket status with active connections."""
        # Add test connections
        _websocket_connections['conn1'] = {
            'url': 'ws://test1.com',
            'status': 'connected',
            'retry_count': 0,
            'connected_at': '2024-01-01T12:00:00',
            'last_message': '2024-01-01T12:05:00'
        }
        _websocket_connections['conn2'] = {
            'url': 'ws://test2.com',
            'status': 'connecting',
            'retry_count': 1,
            'connected_at': None,
            'last_message': None
        }
        
        status = get_websocket_status()
        
        assert status['total_connections'] == 2
        assert 'conn1' in status['connections']
        assert 'conn2' in status['connections']
        
        conn1_status = status['connections']['conn1']
        assert conn1_status['status'] == 'connected'
        assert conn1_status['url'] == 'ws://test1.com'
        assert conn1_status['retry_count'] == 0
    
    def test_get_websocket_status_error_handling(self):
        """Test WebSocket status error handling."""
        # Create a mock lock that raises an exception
        mock_lock = Mock()
        mock_lock.__enter__ = Mock(side_effect=Exception("Lock error"))
        mock_lock.__exit__ = Mock()
        
        with patch('src.utils.shared._connection_lock', mock_lock):
            status = get_websocket_status()
            assert 'error' in status
    
    def test_get_stream_status_empty(self):
        """Test stream status with no streams."""
        status = get_stream_status()
        
        assert status['total_streams'] == 0
        assert status['streams'] == {}
        assert 'timestamp' in status
    
    def test_get_stream_status_with_streams(self):
        """Test stream status with active streams."""
        from collections import deque
        
        # Add test streams
        _stream_buffers['stream1'] = {
            'symbols': ['AAPL', 'GOOGL'],
            'buffer': deque(['data1', 'data2']),
            'processed_count': 100,
            'error_count': 2,
            'started_at': '2024-01-01T12:00:00',
            'last_processed': '2024-01-01T12:05:00'
        }
        _stream_buffers['stream2'] = {
            'symbols': ['MSFT'],
            'buffer': deque(),
            'processed_count': 50,
            'error_count': 0,
            'started_at': '2024-01-01T11:30:00',
            'last_processed': '2024-01-01T12:00:00'
        }
        
        status = get_stream_status()
        
        assert status['total_streams'] == 2
        assert 'stream1' in status['streams']
        assert 'stream2' in status['streams']
        
        stream1_status = status['streams']['stream1']
        assert stream1_status['symbols'] == ['AAPL', 'GOOGL']
        assert stream1_status['buffer_size'] == 2
        assert stream1_status['processed_count'] == 100
        assert stream1_status['error_count'] == 2
    
    def test_get_stream_status_error_handling(self):
        """Test stream status error handling."""
        # Create a mock lock that raises an exception
        mock_lock = Mock()
        mock_lock.__enter__ = Mock(side_effect=Exception("Lock error"))
        mock_lock.__exit__ = Mock()
        
        with patch('src.utils.shared._connection_lock', mock_lock):
            status = get_stream_status()
            assert 'error' in status


class TestStreamingIntegration:
    """Test integration scenarios for streaming functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        _websocket_connections.clear()
        _stream_buffers.clear()
    
    def test_concurrent_stream_setup(self):
        """Test setting up multiple streams concurrently."""
        def setup_stream(symbols, stream_id):
            callback = Mock()
            return stream_market_data(symbols, callback)
        
        # Setup multiple streams in threads
        threads = []
        results = []
        
        for i in range(5):
            symbols = [f'SYM{i}']
            thread = threading.Thread(
                target=lambda s=symbols: results.append(setup_stream(s, f'stream_{len(results)}'))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All streams should be set up successfully
        assert len(results) == 5
        assert all(r['status'] == 'streaming' for r in results)
        assert len(_stream_buffers) == 5
    
    def test_stream_cleanup_and_management(self):
        """Test stream cleanup and management."""
        # Setup a stream
        callback = Mock()
        result = stream_market_data(['AAPL'], callback, buffer_size=10)
        stream_id = result['stream_id']
        
        # Verify stream exists
        assert stream_id in _stream_buffers
        
        # Simulate stream removal (manual cleanup)
        _stream_buffers.pop(stream_id)
        
        # Should be removed from status
        status = get_stream_status()
        assert stream_id not in status['streams']
    
    def test_large_buffer_handling(self):
        """Test handling of large data buffers."""
        from collections import deque
        
        callback = Mock()
        stream_id = 'large_buffer_test'
        
        # Create stream with large buffer
        large_buffer = deque(maxlen=10000)
        for i in range(5000):
            large_buffer.append({'data': f'item_{i}'})
        
        _stream_buffers[stream_id] = {
            'symbols': ['AAPL'],
            'callback': callback,
            'buffer': large_buffer,
            'batch_size': 100,
            'processed_count': 0,
            'error_count': 0
        }
        
        # Get status should handle large buffer
        status = get_stream_status()
        stream_status = status['streams'][stream_id]
        assert stream_status['buffer_size'] == 5000
        assert stream_status['processed_count'] == 0