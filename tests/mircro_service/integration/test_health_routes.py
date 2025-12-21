import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_root(client: AsyncClient):
    """Test GET /health/ returns healthy status"""
    # Act
    response = await client.get("/health/")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_liveness(client: AsyncClient):
    """Test GET /health/live returns alive status"""
    # Act
    response = await client.get("/health/live")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


@pytest.mark.asyncio
async def test_health_readiness(client: AsyncClient):
    """Test GET /health/ready checks database connectivity"""
    # Act
    response = await client.get("/health/ready")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] == "connected"
    assert data["service"] == "Attribute-Prediction API"
