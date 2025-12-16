import pytest
from httpx import AsyncClient
import uuid


@pytest.mark.asyncio
async def test_create_metric_success(client: AsyncClient):
    """Test POST /model-metrics/create successfully creates a metric"""
    # Arrange - Create model type first
    model_type_response = await client.post("/model-types/", json={"name": "GraphSAGE"})
    model_type_id = model_type_response.json()["id"]

    metric_data = {
        "model_type": model_type_id,
        "train_time_min": 45,
        "bias": 0.15,
        "gap": 0.25,
        "path_to_save": "/models/test_model.pth"
    }

    # Act
    response = await client.post("/model-metrics/create", json=metric_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["train_time_min"] == 45
    assert data["bias"] == 0.15
    assert data["gap"] == 0.25


@pytest.mark.asyncio
async def test_create_metric_invalid_model_type(client: AsyncClient):
    """Test POST /model-metrics/create with non-existent model_type returns 400"""
    # Arrange
    metric_data = {
        "model_type": str(uuid.uuid4()),
        "train_time_min": 45,
        "bias": 0.15,
        "gap": 0.25,
        "path_to_save": "/models/test_model.pth"
    }

    # Act
    response = await client.post("/model-metrics/create", json=metric_data)

    # Assert
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_metrics_success(client: AsyncClient):
    """Test GET /model-metrics/{model_type} returns metrics"""
    # Arrange - Create model type and metric
    model_type_response = await client.post("/model-types/", json={"name": "GCN"})
    model_type_id = model_type_response.json()["id"]

    metric_data = {
        "model_type": model_type_id,
        "train_time_min": 30,
        "bias": 0.1,
        "gap": 0.2,
        "path_to_save": "/models/gcn_model.pth"
    }
    await client.post("/model-metrics/create", json=metric_data)

    # Act
    response = await client.get(f"/model-metrics/{model_type_id}")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["train_time_min"] == 30


@pytest.mark.asyncio
async def test_get_metrics_invalid_uuid(client: AsyncClient):
    """Test GET /model-metrics/{model_type} with invalid UUID returns 400"""
    # Act
    response = await client.get("/model-metrics/invalid-uuid")

    # Assert
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_metrics_not_found(client: AsyncClient):
    """Test GET /model-metrics/{model_type} with no metrics returns 404"""
    # Arrange
    model_type_id = str(uuid.uuid4())

    # Act
    response = await client.get(f"/model-metrics/{model_type_id}")

    # Assert
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_hyperparam_success(client: AsyncClient):
    """Test POST /model-metrics/hyperparam/create successfully creates hyperparameter"""
    # Arrange - Create model type and metric
    model_type_response = await client.post("/model-types/", json={"name": "GraphSAGE"})
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

    hyperparam_data = {
        "model_id": model_id,
        "param_name": "learning_rate",
        "param_value": "0.001"
    }

    # Act
    response = await client.post("/model-metrics/hyperparam/create", json=hyperparam_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["param_name"] == "learning_rate"
    assert data["param_value"] == "0.001"


@pytest.mark.asyncio
async def test_create_hyperparam_invalid_model_id(client: AsyncClient):
    """Test POST /model-metrics/hyperparam/create with invalid model_id returns 400"""
    # Arrange
    hyperparam_data = {
        "model_id": str(uuid.uuid4()),
        "param_name": "learning_rate",
        "param_value": "0.001"
    }

    # Act
    response = await client.post("/model-metrics/hyperparam/create", json=hyperparam_data)

    # Assert
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_create_loss_success(client: AsyncClient):
    """Test POST /model-metrics/loss/create successfully creates loss record"""
    # Arrange - Create model type and metric
    model_type_response = await client.post("/model-types/", json={"name": "GCN"})
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

    loss_data = {
        "model_id": model_id,
        "type": "training",
        "loss_value": 0.45,
        "loss_unit": "MSE"
    }

    # Act
    response = await client.post("/model-metrics/loss/create", json=loss_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["type"] == "training"
    assert data["loss_value"] == 0.45
    assert data["loss_unit"] == "MSE"


@pytest.mark.asyncio
async def test_create_loss_invalid_model_id(client: AsyncClient):
    """Test POST /model-metrics/loss/create with invalid model_id returns 400"""
    # Arrange
    loss_data = {
        "model_id": str(uuid.uuid4()),
        "type": "training",
        "loss_value": 0.45,
        "loss_unit": "MSE"
    }

    # Act
    response = await client.post("/model-metrics/loss/create", json=loss_data)

    # Assert
    assert response.status_code == 400
