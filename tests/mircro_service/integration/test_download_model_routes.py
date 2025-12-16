import pytest
from httpx import AsyncClient
import uuid
from unittest.mock import patch
import tempfile
import os


@pytest.mark.asyncio
async def test_download_model_invalid_uuid(client: AsyncClient):
    """Test GET /download_model/{model_id} with invalid UUID returns 400"""
    # Act
    response = await client.get("/download_model/invalid-uuid")

    # Assert
    assert response.status_code == 400
    assert "Invalid UUID" in response.json()["detail"]


@pytest.mark.asyncio
async def test_download_model_not_found(client: AsyncClient):
    """Test GET /download_model/{model_id} with non-existent model returns 404"""
    # Arrange
    model_id = str(uuid.uuid4())

    # Act
    response = await client.get(f"/download_model/{model_id}")

    # Assert
    assert response.status_code == 404


@pytest.mark.asyncio
@patch('src.app.services.download_model_utils.os.path.isfile')
async def test_download_model_file_not_exists(mock_isfile, client: AsyncClient):
    """Test GET /download_model/{model_id} when file doesn't exist on disk returns 404"""
    # Arrange - Create model type and metric
    model_type_response = await client.post("/model-types/create", json={"name": "GraphSAGE"})
    model_type_id = model_type_response.json()["id"]

    metric_data = {
        "model_type": model_type_id,
        "train_time_min": 45,
        "bias": 0.15,
        "gap": 0.25,
        "path_to_save": "/nonexistent/model.pth"
    }
    metric_response = await client.post("/model-metrics/create", json=metric_data)
    model_id = metric_response.json()["id"]

    # Mock file not existing
    mock_isfile.return_value = False

    # Act
    response = await client.get(f"/download_model/{model_id}")

    # Assert
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_download_model_success(client: AsyncClient):
    """Test GET /download_model/{model_id} successfully returns file"""
    # Arrange - Create a temporary file with mock model data
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pth', delete=False) as tmp_file:
        tmp_file.write(b"mock pytorch model data")
        tmp_path = tmp_file.name

    try:
        # Create model type and metric pointing to the temp file
        model_type_response = await client.post("/model-types/create", json={"name": "GCN"})
        model_type_id = model_type_response.json()["id"]

        metric_data = {
            "model_type": model_type_id,
            "train_time_min": 30,
            "bias": 0.1,
            "gap": 0.2,
            "path_to_save": tmp_path
        }
        metric_response = await client.post("/model-metrics/create", json=metric_data)
        model_id = metric_response.json()["id"]

        # Act
        response = await client.get(f"/download_model/{model_id}")

        # Assert
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert b"mock pytorch model data" in response.content
    finally:
        # Cleanup - Remove temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
