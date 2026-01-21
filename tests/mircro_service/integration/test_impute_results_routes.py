import pytest
from httpx import AsyncClient
import uuid
import time
import shutil
from pathlib import Path
from src.app.config import settings


@pytest.fixture(autouse=True)
def clean_csv_files():
    """Clean up CSV files before and after each test"""
    csv_dir = settings.impute_results_dir
    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup after test
    if csv_dir.exists():
        shutil.rmtree(csv_dir)


@pytest.mark.asyncio
async def test_create_impute_result_success(client: AsyncClient):
    """Test POST /impute-result/ successfully creates an impute result"""
    # Arrange - Use a valid UUID (no need to create model in DB anymore)
    model_id = str(uuid.uuid4())

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
async def test_create_impute_result_invalid_uuid(client: AsyncClient):
    """Test POST /impute-result/ with invalid UUID returns 422 (Pydantic validation)"""
    # Arrange
    impute_data = {
        "model_id": "invalid-uuid",
        "road_id": "123",
        "tms": int(time.time()),
        "value": 50.5,
        "imputed": True
    }

    # Act
    response = await client.post("/impute-result/", json=impute_data)

    # Assert - Pydantic validation returns 422
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_impute_results_success(client: AsyncClient):
    """Test GET /impute-result/{model_id}/{road_id}/{start_time}/{end_time} returns results"""
    # Arrange - Create impute results
    model_id = str(uuid.uuid4())

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
async def test_get_impute_results_filtered_by_road(client: AsyncClient):
    """Test GET /impute-result/ filters results by road_id"""
    # Arrange
    model_id = str(uuid.uuid4())
    start_time = int(time.time())

    # Create results for multiple roads
    for road_id in ["111", "222"]:
        for i in range(2):
            impute_data = {
                "model_id": model_id,
                "road_id": road_id,
                "tms": start_time + i * 60,
                "value": 30.0 + float(road_id) + i,
                "imputed": False
            }
            await client.post("/impute-result/", json=impute_data)

    end_time = start_time + 180

    # Act - Get only results for road 111
    response = await client.get(f"/impute-result/{model_id}/111/{start_time}/{end_time}")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    # Verify all results are for road 111 (indirectly by checking count)


@pytest.mark.asyncio
async def test_get_road_ids_success(client: AsyncClient):
    """Test GET /impute-result/roads/{model_id} returns all road IDs"""
    # Arrange - Create impute results for multiple roads
    model_id = str(uuid.uuid4())

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
    # Arrange - Create multiple impute results with different timestamps
    model_id = str(uuid.uuid4())

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


@pytest.mark.asyncio
async def test_create_impute_result_duplicate(client: AsyncClient):
    """Test POST /impute-result/ with duplicate updates existing entry"""
    # Arrange
    model_id = str(uuid.uuid4())

    impute_data_1 = {
        "model_id": model_id,
        "road_id": "555",
        "tms": 1000,
        "value": 50.5,
        "imputed": False
    }

    impute_data_2 = {
        "model_id": model_id,
        "road_id": "555",
        "tms": 1000,  # Same timestamp
        "value": 75.0,  # Different value
        "imputed": True  # Different imputed
    }

    # Act
    response1 = await client.post("/impute-result/", json=impute_data_1)
    response2 = await client.post("/impute-result/", json=impute_data_2)

    # Assert
    assert response1.status_code == 201
    assert response2.status_code == 201
    assert response2.json()["value"] == 75.0
    assert response2.json()["imputed"] is True
