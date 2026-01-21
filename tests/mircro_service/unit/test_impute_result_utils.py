import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import uuid
from pathlib import Path
from src.app.services.impute_result_utils import (
    find_impute_results,
    find_road_ids,
    find_timespan,
    create_impute_result,
    _read_csv,
    _write_csv,
    _get_csv_path
)
from src.app.schemas import ImputeResultCreate
from src.app.exceptions import InvalidUUIDException, NotFoundException


@pytest.mark.asyncio
async def test_find_impute_results_success():
    """Test finding impute results successfully"""
    # Arrange
    model_id = str(uuid.uuid4())
    road_id = "1"

    mock_df = pd.DataFrame({
        'road_id': ['1', '1', '2'],
        'tms': [1000, 2000, 1500],
        'value': [50.5, 55.0, 60.0],
        'imputed': [False, True, False]
    })

    with patch('src.app.services.impute_result_utils._read_csv', return_value=mock_df):
        # Act
        result = await find_impute_results(model_id, road_id, 0, 3000)

        # Assert
        assert len(result) == 2
        assert result[0].tms == 1000
        assert result[0].value == 50.5
        assert result[0].imputed is False
        assert result[1].tms == 2000


@pytest.mark.asyncio
async def test_find_impute_results_invalid_uuid():
    """Test finding impute results with invalid UUID"""
    # Act & Assert
    with pytest.raises(InvalidUUIDException):
        await find_impute_results("invalid-uuid", "1", 0, 1000)


@pytest.mark.asyncio
async def test_find_impute_results_not_found():
    """Test finding impute results when none exist"""
    # Arrange
    model_id = str(uuid.uuid4())

    with patch('src.app.services.impute_result_utils._read_csv', return_value=None):
        # Act & Assert
        with pytest.raises(NotFoundException):
            await find_impute_results(model_id, "1", 0, 1000)


@pytest.mark.asyncio
async def test_find_impute_results_road_not_found():
    """Test finding impute results when road doesn't exist in CSV"""
    # Arrange
    model_id = str(uuid.uuid4())

    mock_df = pd.DataFrame({
        'road_id': ['2', '3'],
        'tms': [1000, 2000],
        'value': [50.5, 55.0],
        'imputed': [False, True]
    })

    with patch('src.app.services.impute_result_utils._read_csv', return_value=mock_df):
        # Act & Assert
        with pytest.raises(NotFoundException):
            await find_impute_results(model_id, "1", 0, 3000)


@pytest.mark.asyncio
async def test_find_road_ids_success():
    """Test finding road IDs for a model"""
    # Arrange
    model_id = str(uuid.uuid4())

    mock_df = pd.DataFrame({
        'road_id': ['1', '2', '1', '3'],
        'tms': [1000, 2000, 1500, 3000],
        'value': [50.5, 55.0, 60.0, 65.0],
        'imputed': [False, True, False, True]
    })

    with patch('src.app.services.impute_result_utils._read_csv', return_value=mock_df):
        # Act
        result = await find_road_ids(model_id)

        # Assert
        assert len(result) == 3
        road_ids = [r.road_id for r in result]
        assert '1' in road_ids
        assert '2' in road_ids
        assert '3' in road_ids


@pytest.mark.asyncio
async def test_find_road_ids_invalid_uuid():
    """Test finding road IDs with invalid UUID"""
    # Act & Assert
    with pytest.raises(InvalidUUIDException):
        await find_road_ids("invalid-uuid")


@pytest.mark.asyncio
async def test_find_road_ids_not_found():
    """Test finding road IDs when none exist"""
    # Arrange
    model_id = str(uuid.uuid4())

    with patch('src.app.services.impute_result_utils._read_csv', return_value=None):
        # Act & Assert
        with pytest.raises(NotFoundException):
            await find_road_ids(model_id)


@pytest.mark.asyncio
async def test_find_timespan_success():
    """Test finding timespan for a model and road"""
    # Arrange
    model_id = str(uuid.uuid4())
    road_id = "1"

    mock_df = pd.DataFrame({
        'road_id': ['1', '1', '2'],
        'tms': [1000, 5000, 3000],
        'value': [50.5, 55.0, 60.0],
        'imputed': [False, True, False]
    })

    with patch('src.app.services.impute_result_utils._read_csv', return_value=mock_df):
        # Act
        result = await find_timespan(model_id, road_id)

        # Assert
        assert result.start_time == 1000
        assert result.end_time == 5000


@pytest.mark.asyncio
async def test_find_timespan_invalid_uuid():
    """Test finding timespan with invalid UUID"""
    # Act & Assert
    with pytest.raises(InvalidUUIDException):
        await find_timespan("invalid-uuid", "1")


@pytest.mark.asyncio
async def test_find_timespan_no_data():
    """Test finding timespan when no data exists"""
    # Arrange
    model_id = str(uuid.uuid4())

    with patch('src.app.services.impute_result_utils._read_csv', return_value=None):
        # Act & Assert
        with pytest.raises(NotFoundException):
            await find_timespan(model_id, "1")


@pytest.mark.asyncio
async def test_find_timespan_road_not_found():
    """Test finding timespan when road doesn't exist"""
    # Arrange
    model_id = str(uuid.uuid4())

    mock_df = pd.DataFrame({
        'road_id': ['2', '3'],
        'tms': [1000, 2000],
        'value': [50.5, 55.0],
        'imputed': [False, True]
    })

    with patch('src.app.services.impute_result_utils._read_csv', return_value=mock_df):
        # Act & Assert
        with pytest.raises(NotFoundException):
            await find_timespan(model_id, "1")


@pytest.mark.asyncio
async def test_create_impute_result_success():
    """Test creating an impute result successfully"""
    # Arrange
    result_data = ImputeResultCreate(
        model_id=uuid.uuid4(),
        road_id="1",
        tms=1000,
        value=50.5,
        imputed=False
    )

    mock_df = pd.DataFrame({
        'road_id': ['2'],
        'tms': [2000],
        'value': [60.0],
        'imputed': [True]
    })

    with patch('src.app.services.impute_result_utils._read_csv', return_value=mock_df) as mock_read, \
         patch('src.app.services.impute_result_utils._write_csv') as mock_write:

        # Act
        result = await create_impute_result(result_data)

        # Assert
        assert result.tms == result_data.tms
        assert result.value == result_data.value
        assert result.imputed == result_data.imputed
        mock_write.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_impute_result_new_file():
    """Test creating impute result when CSV doesn't exist yet"""
    # Arrange
    result_data = ImputeResultCreate(
        model_id=uuid.uuid4(),
        road_id="1",
        tms=1000,
        value=50.5,
        imputed=False
    )

    with patch('src.app.services.impute_result_utils._read_csv', return_value=None) as mock_read, \
         patch('src.app.services.impute_result_utils._write_csv') as mock_write:

        # Act
        result = await create_impute_result(result_data)

        # Assert
        assert result.tms == result_data.tms
        mock_write.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_impute_result_duplicate_update():
    """Test creating impute result with duplicate updates existing entry"""
    # Arrange
    result_data = ImputeResultCreate(
        model_id=uuid.uuid4(),
        road_id="1",
        tms=1000,
        value=75.0,  # Updated value
        imputed=True  # Updated imputed flag
    )

    # Existing data has same road_id and tms
    mock_df = pd.DataFrame({
        'road_id': ['1', '2'],
        'tms': [1000, 2000],
        'value': [50.5, 60.0],
        'imputed': [False, True]
    })

    with patch('src.app.services.impute_result_utils._read_csv', return_value=mock_df) as mock_read, \
         patch('src.app.services.impute_result_utils._write_csv') as mock_write:

        # Act
        result = await create_impute_result(result_data)

        # Assert
        assert result.value == 75.0
        assert result.imputed is True
        mock_write.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_impute_result_invalid_uuid():
    """Test creating impute result with invalid UUID"""
    # Arrange & Act & Assert
    # Pydantic will raise ValidationError when creating ImputeResultCreate with invalid UUID
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        result_data = ImputeResultCreate(
            model_id="invalid-uuid",  # type: ignore
            road_id="1",
            tms=1000,
            value=50.5,
            imputed=False
        )
