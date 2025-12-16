import pytest
from httpx import AsyncClient
import uuid
import time


@pytest.mark.asyncio
async def test_create_impute_result_success(client: AsyncClient):
    """Test POST /impute-result/ successfully creates an impute result"""
    # Arrange - Create model type and metric first
    model_type_response = await client.post("/model-types/create", json={"name": "GraphSAGE"})
    model_type_id = model_type_response.json()["id"]

    metric_data = {
        "model_type": model_type_id,
        "train_time_min": 45,
        "bias": 0.15,
        "gap": 0.25,
        "path_to_save": "/models/test_model.pth"
    }
    metric_response = await client.post("/model-metrics/create", json=metric_data)
    model_id = metric_response.json()["id"]

    impute_data = {
        "model_id": model_id,
        "road_id": "123",
        "tms": int(time.time()),
        "value": 50.5,
        "imputed": True
    }

    # Act
    response = await client.post("/impute-result/", json=impute_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["tms"] == impute_data["tms"]
    assert data["value"] == 50.5
    assert data["imputed"] is True


@pytest.mark.asyncio
async def test_create_impute_result_invalid_model_id(client: AsyncClient):
    """Test POST /impute-result/ with non-existent model_id returns 400"""
    # Arrange
    impute_data = {
        "model_id": str(uuid.uuid4()),
        "road_id": "123",
        "tms": int(time.time()),
        "value": 50.5,
        "imputed": True
    }

    # Act
    response = await client.post("/impute-result/", json=impute_data)

    # Assert
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_impute_results_success(client: AsyncClient):
    """Test GET /impute-result/{model_id}/{road_id}/{start_time}/{end_time} returns results"""
    # Arrange - Create model, metric, and impute results
    model_type_response = await client.post("/model-types/create", json={"name": "GCN"})
    model_type_id = model_type_response.json()["id"]

    metric_data = {
        "model_type": model_type_id,
        "train_time_min": 30,
        "bias": 0.1,
        "gap": 0.2,
        "path_to_save": "/models/gcn_model.pth"
    }
    metric_response = await client.post("/model-metrics/create", json=metric_data)
    model_id = metric_response.json()["id"]

    # Create multiple impute results
    start_time = int(time.time())
    for i in range(3):
        impute_data = {
            "model_id": model_id,
            "road_id": "456",
            "tms": start_time + i * 60,
            "value": 40.0 + i * 5.0,
            "imputed": False
        }
        await client.post("/impute-result/", json=impute_data)

    end_time = start_time + 180

    # Act
    response = await client.get(f"/impute-result/{model_id}/456/{start_time}/{end_time}")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["value"] == 40.0


@pytest.mark.asyncio
async def test_get_impute_results_invalid_uuid(client: AsyncClient):
    """Test GET /impute-result/ with invalid UUID returns 400"""
    # Act
    response = await client.get("/impute-result/invalid-uuid/123/1000/2000")

    # Assert
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_impute_results_not_found(client: AsyncClient):
    """Test GET /impute-result/ with no results returns 404"""
    # Arrange
    model_id = str(uuid.uuid4())
    start_time = int(time.time())
    end_time = start_time + 1000

    # Act
    response = await client.get(f"/impute-result/{model_id}/999/{start_time}/{end_time}")

    # Assert
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_road_ids_success(client: AsyncClient):
    """Test GET /impute-result/roads/{model_id} returns all road IDs"""
    # Arrange - Create model and impute results for multiple roads
    model_type_response = await client.post("/model-types/create", json={"name": "GraphSAGE"})
    model_type_id = model_type_response.json()["id"]

    metric_data = {
        "model_type": model_type_id,
        "train_time_min": 45,
        "bias": 0.15,
        "gap": 0.25,
        "path_to_save": "/models/test_model.pth"
    }
    metric_response = await client.post("/model-metrics/create", json=metric_data)
    model_id = metric_response.json()["id"]

    # Create results for different roads
    for road_id in ["101", "102", "103"]:
        impute_data = {
            "model_id": model_id,
            "road_id": road_id,
            "tms": int(time.time()),
            "value": 50.5,
            "imputed": True
        }
        await client.post("/impute-result/", json=impute_data)

    # Act
    response = await client.get(f"/impute-result/roads/{model_id}")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    road_ids = [item["road_id"] for item in data]
    assert "101" in road_ids
    assert "102" in road_ids
    assert "103" in road_ids


@pytest.mark.asyncio
async def test_get_road_ids_invalid_uuid(client: AsyncClient):
    """Test GET /impute-result/roads/{model_id} with invalid UUID returns 400"""
    # Act
    response = await client.get("/impute-result/roads/invalid-uuid")

    # Assert
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_road_ids_not_found(client: AsyncClient):
    """Test GET /impute-result/roads/{model_id} with no roads returns 404"""
    # Arrange
    model_id = str(uuid.uuid4())

    # Act
    response = await client.get(f"/impute-result/roads/{model_id}")

    # Assert
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_time_interval_success(client: AsyncClient):
    """Test GET /impute-result/time-interval/{model_id}/{road_id} returns time range"""
    # Arrange - Create model and multiple impute results with different timestamps
    model_type_response = await client.post("/model-types/create", json={"name": "GCN"})
    model_type_id = model_type_response.json()["id"]

    metric_data = {
        "model_type": model_type_id,
        "train_time_min": 30,
        "bias": 0.1,
        "gap": 0.2,
        "path_to_save": "/models/gcn_model.pth"
    }
    metric_response = await client.post("/model-metrics/create", json=metric_data)
    model_id = metric_response.json()["id"]

    base_time = int(time.time())
    timestamps = [base_time, base_time + 300, base_time + 600]

    for tms in timestamps:
        impute_data = {
            "model_id": model_id,
            "road_id": "789",
            "tms": tms,
            "value": 55.0,
            "imputed": False
        }
        await client.post("/impute-result/", json=impute_data)

    # Act
    response = await client.get(f"/impute-result/time-interval/{model_id}/789")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "start_time" in data
    assert "end_time" in data
    assert data["start_time"] == base_time
    assert data["end_time"] == base_time + 600


@pytest.mark.asyncio
async def test_get_time_interval_invalid_uuid(client: AsyncClient):
    """Test GET /impute-result/time-interval/ with invalid UUID returns 400"""
    # Act
    response = await client.get("/impute-result/time-interval/invalid-uuid/123")

    # Assert
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_time_interval_not_found(client: AsyncClient):
    """Test GET /impute-result/time-interval/ with no data returns 404"""
    # Arrange
    model_id = str(uuid.uuid4())

    # Act
    response = await client.get(f"/impute-result/time-interval/{model_id}/999")

    # Assert
    assert response.status_code == 404
