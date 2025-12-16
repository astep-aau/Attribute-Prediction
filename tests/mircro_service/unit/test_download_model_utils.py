import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.app.services.download_model_utils import (
    build_file_response,
    get_model_path,
    get_file_name
)
from src.app.exceptions import InvalidUUIDException, NotFoundException
import uuid
import os


@pytest.mark.asyncio
async def test_get_model_path_success():
    """Test getting model path successfully"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())
    expected_path = "/path/to/model.pth"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = expected_path
    mock_db.execute.return_value = mock_result

    # Act
    with patch('os.path.isfile', return_value=True):
        result = await get_model_path(model_id, mock_db)

    # Assert
    assert result == expected_path
    mock_db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_model_path_invalid_uuid():
    """Test getting model path with invalid UUID"""
    # Arrange
    mock_db = AsyncMock()

    # Act & Assert
    with pytest.raises(InvalidUUIDException):
        await get_model_path("invalid-uuid", mock_db)


@pytest.mark.asyncio
async def test_get_model_path_not_found_in_db():
    """Test getting model path when not in database"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    # Act & Assert
    with pytest.raises(NotFoundException):
        await get_model_path(model_id, mock_db)


@pytest.mark.asyncio
async def test_get_model_path_file_not_exists():
    """Test getting model path when file doesn't exist on disk"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())
    path = "/nonexistent/model.pth"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = path
    mock_db.execute.return_value = mock_result

    # Act & Assert
    with patch('os.path.isfile', return_value=False):
        with pytest.raises(NotFoundException):
            await get_model_path(model_id, mock_db)


def test_get_file_name_with_extension():
    """Test extracting filename and changing extension"""
    # Arrange
    filepath = "/path/to/model.pt"

    # Act
    result = get_file_name(filepath)

    # Assert
    assert result == "model.pth"


def test_get_file_name_already_pth():
    """Test filename that already has .pth extension"""
    # Arrange
    filepath = "/path/to/model.pth"

    # Act
    result = get_file_name(filepath)

    # Assert
    assert result == "model.pth"


def test_get_file_name_no_extension():
    """Test filename without extension"""
    # Arrange
    filepath = "/path/to/model"

    # Act
    result = get_file_name(filepath)

    # Assert
    assert result == "model.pth"


@pytest.mark.asyncio
async def test_build_file_response_success():
    """Test building file response successfully"""
    # Arrange
    mock_db = AsyncMock()
    model_id = str(uuid.uuid4())
    expected_path = "/path/to/model.pth"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = expected_path
    mock_db.execute.return_value = mock_result

    # Act
    with patch('os.path.isfile', return_value=True):
        result = await build_file_response(model_id, mock_db)

    # Assert
    assert result.path == expected_path
    assert result.filename == "model.pth"
    assert result.media_type == "application/octet-stream"
