import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.exc import IntegrityError
from src.app.services.impute_result_utils import (
    find_impute_results,
    find_road_ids,
    find_timespan,
    create_impute_result
)
from src.app.schemas import ImputeResultCreate
from src.app.exceptions import InvalidUUIDException, NotFoundException, ForeignKeyViolationException
import uuid


@pytest.mark.asyncio
async def test_find_impute_results_success():
    """Test finding impute results successfully"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())
    road_id = 1

    mock_result = MagicMock()
    mock_result.all.return_value = [
        (1000, 50.5, False),
        (2000, 55.0, True)
    ]
    mock_db.execute.return_value = mock_result

    # Act
    result = await find_impute_results(model_id, road_id, 0, 3000, mock_db)

    # Assert
    assert len(result) == 2
    assert result[0].tms == 1000
    assert result[0].value == 50.5
    assert result[0].imputed is False
    mock_db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_find_impute_results_invalid_uuid():
    """Test finding impute results with invalid UUID"""
    # Arrange
    mock_db = AsyncMock()

    # Act & Assert
    with pytest.raises(InvalidUUIDException):
        await find_impute_results("invalid-uuid", 1, 0, 1000, mock_db)


@pytest.mark.asyncio
async def test_find_impute_results_not_found():
    """Test finding impute results when none exist"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_db.execute.return_value = mock_result

    # Act & Assert
    with pytest.raises(NotFoundException):
        await find_impute_results(model_id, 1, 0, 1000, mock_db)


@pytest.mark.asyncio
async def test_find_road_ids_success():
    """Test finding road IDs for a model"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = ["1", "2", "3"]
    mock_db.execute.return_value = mock_result

    # Act
    result = await find_road_ids(model_id, mock_db)

    # Assert
    assert len(result) == 3
    assert result[0].road_id == "1"
    assert result[1].road_id == "2"
    mock_db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_find_road_ids_invalid_uuid():
    """Test finding road IDs with invalid UUID"""
    # Arrange
    mock_db = AsyncMock()

    # Act & Assert
    with pytest.raises(InvalidUUIDException):
        await find_road_ids("invalid-uuid", mock_db)


@pytest.mark.asyncio
async def test_find_road_ids_not_found():
    """Test finding road IDs when none exist"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    # Act & Assert
    with pytest.raises(NotFoundException):
        await find_road_ids(model_id, mock_db)


@pytest.mark.asyncio
async def test_find_timespan_success():
    """Test finding timespan for a model and road"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())
    road_id = "1"

    mock_result = MagicMock()
    mock_result.one.return_value = (1000, 5000)
    mock_db.execute.return_value = mock_result

    # Act
    result = await find_timespan(model_id, road_id, mock_db)

    # Assert
    assert result.start_time == 1000
    assert result.end_time == 5000
    mock_db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_find_timespan_invalid_uuid():
    """Test finding timespan with invalid UUID"""
    # Arrange
    mock_db = AsyncMock()

    # Act & Assert
    with pytest.raises(InvalidUUIDException):
        await find_timespan("invalid-uuid", "1", mock_db)


@pytest.mark.asyncio
async def test_find_timespan_no_min():
    """Test finding timespan when min time is None"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.one.return_value = (None, 5000)
    mock_db.execute.return_value = mock_result

    # Act & Assert
    with pytest.raises(NotFoundException):
        await find_timespan(model_id, "1", mock_db)


@pytest.mark.asyncio
async def test_find_timespan_no_max():
    """Test finding timespan when max time is None"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.one.return_value = (1000, None)
    mock_db.execute.return_value = mock_result

    # Act & Assert
    with pytest.raises(NotFoundException):
        await find_timespan(model_id, "1", mock_db)


@pytest.mark.asyncio
async def test_create_impute_result_success():
    """Test creating an impute result successfully"""
    # Arrange
    mock_db = AsyncMock()
    result_data = ImputeResultCreate(
        model_id=uuid.uuid4(),
        road_id="1",
        tms=1000,
        value=50.5,
        imputed=False
    )

    # Act
    result = await create_impute_result(result_data, mock_db)

    # Assert
    assert result.model_id == result_data.model_id
    assert result.road_id == result_data.road_id
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_impute_result_foreign_key_violation():
    """Test creating impute result with invalid model_id"""
    # Arrange
    mock_db = AsyncMock()
    mock_db.commit.side_effect = IntegrityError("statement", "params", "foreign key")
    result_data = ImputeResultCreate(
        model_id=uuid.uuid4(),
        road_id="1",
        tms=1000,
        value=50.5,
        imputed=False
    )

    # Act & Assert
    with pytest.raises(ForeignKeyViolationException):
        await create_impute_result(result_data, mock_db)

    mock_db.rollback.assert_awaited_once()
