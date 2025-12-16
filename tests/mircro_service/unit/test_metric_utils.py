import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.exc import IntegrityError
from src.app.services.metric_utils import (
    find_metric,
    find_hyperparams,
    find_loss,
    create_metric,
    create_hyperparam,
    create_loss
)
from src.app.schemas import ModelMetricsCreate, Hyperparam, ModelLoss
from src.app.exceptions import InvalidUUIDException, NotFoundException, ForeignKeyViolationException
import uuid


@pytest.mark.asyncio
async def test_find_metric_success():
    """Test finding metrics for a valid model type"""
    # Arrange
    mock_db = AsyncMock()
    model_type_uuid = uuid.uuid4()
    model_id = uuid.uuid4()

    # Mock hyperparam
    mock_hyperparam = MagicMock()
    mock_hyperparam.model_id = model_id
    mock_hyperparam.param_name = "learning_rate"
    mock_hyperparam.param_value = "0.001"

    # Mock loss
    mock_loss = MagicMock()
    mock_loss.model_id = model_id
    mock_loss.type = "training"
    mock_loss.loss_value = 0.5
    mock_loss.loss_unit = "MSE"

    mock_metric = MagicMock()
    mock_metric.id = model_id
    mock_metric.model_type = model_type_uuid
    mock_metric.train_time_min = 10
    mock_metric.bias = 0.1
    mock_metric.gap = 0.2

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_metric]
    mock_db.execute.return_value = mock_result

    # Mock find_hyperparams and find_loss to return proper objects
    with patch('src.app.services.metric_utils.find_hyperparams', return_value=[mock_hyperparam]):
        with patch('src.app.services.metric_utils.find_loss', return_value=[mock_loss]):
            # Act
            result = await find_metric(str(model_type_uuid), mock_db)

            # Assert
            assert len(result) == 1
            assert result[0].train_time_min == 10
            mock_db.execute.assert_awaited()


@pytest.mark.asyncio
async def test_find_metric_invalid_uuid():
    """Test finding metrics with invalid UUID raises exception"""
    # Arrange
    mock_db = AsyncMock()

    # Act & Assert
    with pytest.raises(InvalidUUIDException):
        await find_metric("invalid-uuid", mock_db)


@pytest.mark.asyncio
async def test_find_metric_not_found():
    """Test finding metrics when no metrics exist"""
    # Arrange
    mock_db = AsyncMock()
    model_type_uuid = uuid.uuid4()

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    # Act & Assert
    with pytest.raises(NotFoundException):
        await find_metric(str(model_type_uuid), mock_db)


@pytest.mark.asyncio
async def test_find_hyperparams_success():
    """Test finding hyperparameters for a model"""
    # Arrange
    mock_db = AsyncMock()
    model_id = uuid.uuid4()

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [MagicMock(), MagicMock()]
    mock_db.execute.return_value = mock_result

    # Act
    result = await find_hyperparams(model_id, mock_db)

    # Assert
    assert len(result) == 2
    mock_db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_find_loss_success():
    """Test finding loss records for a model"""
    # Arrange
    mock_db = AsyncMock()
    model_id = uuid.uuid4()

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [MagicMock()]
    mock_db.execute.return_value = mock_result

    # Act
    result = await find_loss(model_id, mock_db)

    # Assert
    assert len(result) == 1
    mock_db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_metric_success():
    """Test creating a metric successfully"""
    # Arrange
    mock_db = AsyncMock()
    metric_data = ModelMetricsCreate(
        model_type=uuid.uuid4(),
        train_time_min=10,
        bias=0.1,
        gap=0.2,
        path_to_save="/path/to/model.pth"
    )

    # Act
    result = await create_metric(metric_data, mock_db)

    # Assert
    assert result.model_type == metric_data.model_type
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()
    mock_db.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_metric_foreign_key_violation():
    """Test creating metric with invalid model_type"""
    # Arrange
    mock_db = AsyncMock()
    mock_db.commit.side_effect = IntegrityError("statement", "params", "foreign key")
    metric_data = ModelMetricsCreate(
        model_type=uuid.uuid4(),
        train_time_min=10,
        bias=0.1,
        gap=0.2,
        path_to_save="/path/to/model.pth"
    )

    # Act & Assert
    with pytest.raises(ForeignKeyViolationException):
        await create_metric(metric_data, mock_db)

    mock_db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_hyperparam_success():
    """Test creating a hyperparameter successfully"""
    # Arrange
    mock_db = AsyncMock()
    hyperparam_data = Hyperparam(
        model_id=uuid.uuid4(),
        param_name="learning_rate",
        param_value="0.001"
    )

    # Act
    result = await create_hyperparam(hyperparam_data, mock_db)

    # Assert
    assert result.model_id == hyperparam_data.model_id
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_hyperparam_foreign_key_violation():
    """Test creating hyperparam with invalid model_id"""
    # Arrange
    mock_db = AsyncMock()
    mock_db.commit.side_effect = IntegrityError("statement", "params", "foreign key constraint")
    hyperparam_data = Hyperparam(
        model_id=uuid.uuid4(),
        param_name="learning_rate",
        param_value="0.001"
    )

    # Act & Assert
    with pytest.raises(ForeignKeyViolationException):
        await create_hyperparam(hyperparam_data, mock_db)

    mock_db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_hyperparam_other_integrity_error():
    """Test creating hyperparam with non-foreign-key integrity error"""
    # Arrange
    mock_db = AsyncMock()
    mock_db.commit.side_effect = IntegrityError("statement", "params", "unique constraint violated")
    hyperparam_data = Hyperparam(
        model_id=uuid.uuid4(),
        param_name="learning_rate",
        param_value="0.001"
    )

    # Act & Assert
    with pytest.raises(IntegrityError):
        await create_hyperparam(hyperparam_data, mock_db)

    mock_db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_loss_success():
    """Test creating a loss record successfully"""
    # Arrange
    mock_db = AsyncMock()
    loss_data = ModelLoss(
        model_id=uuid.uuid4(),
        type="training",
        loss_value=0.5,
        loss_unit="MSE"
    )

    # Act
    result = await create_loss(loss_data, mock_db)

    # Assert
    assert result.model_id == loss_data.model_id
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_loss_foreign_key_violation():
    """Test creating loss with invalid model_id"""
    # Arrange
    mock_db = AsyncMock()
    mock_db.commit.side_effect = IntegrityError("statement", "params", "foreign key constraint")
    loss_data = ModelLoss(
        model_id=uuid.uuid4(),
        type="training",
        loss_value=0.5,
        loss_unit="MSE"
    )

    # Act & Assert
    with pytest.raises(ForeignKeyViolationException):
        await create_loss(loss_data, mock_db)

    mock_db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_loss_other_integrity_error():
    """Test creating loss with non-foreign-key integrity error"""
    # Arrange
    mock_db = AsyncMock()
    mock_db.commit.side_effect = IntegrityError("statement", "params", "unique constraint violated")
    loss_data = ModelLoss(
        model_id=uuid.uuid4(),
        type="training",
        loss_value=0.5,
        loss_unit="MSE"
    )

    # Act & Assert
    with pytest.raises(IntegrityError):
        await create_loss(loss_data, mock_db)

    mock_db.rollback.assert_awaited_once()
