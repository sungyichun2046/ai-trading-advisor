"""Tests for main FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestMainEndpoints:
    """Test main application endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AI Trading Advisor API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_nonexistent_endpoint(self):
        """Test accessing non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
