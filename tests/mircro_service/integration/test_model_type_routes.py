import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_model_type_success(client: AsyncClient, sample_model_type_data):
    """Test POST /model-types/ creates a model type successfully"""
    # Act
    response = await client.post("/model-types/create", json=sample_model_type_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == sample_model_type_data["name"]
    assert "id" in data


@pytest.mark.asyncio
async def test_create_model_type_invalid_data(client: AsyncClient):
    """Test POST /model-types/ with invalid data returns 422"""
    # Act
    response = await client.post("/model-types/create", json={})

    # Assert
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_all_model_types_empty(client: AsyncClient):
    """Test GET /model-types/ returns empty list when no data"""
    # Act
    response = await client.get("/model-types/")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


@pytest.mark.asyncio
async def test_get_all_model_types_with_data(client: AsyncClient):
    """Test GET /model-types/ returns all model types"""
    # Arrange - Create some model types
    await client.post("/model-types/create", json={"name": "GraphSAGE"})
    await client.post("/model-types/create", json={"name": "GCN"})

    # Act
    response = await client.get("/model-types/")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "GraphSAGE"
    assert data[1]["name"] == "GCN"


@pytest.mark.asyncio
async def test_create_duplicate_model_type(client: AsyncClient):
    """Test POST /model-types/create with duplicate name returns 400"""
    # Arrange - Create first model type
    await client.post("/model-types/create", json={"name": "GraphSAGE"})

    # Act - Try to create duplicate
    response = await client.post("/model-types/create", json={"name": "GraphSAGE"})

    # Assert
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "constraint" in detail or "unique" in detail
