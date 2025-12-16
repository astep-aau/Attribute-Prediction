import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.exc import IntegrityError
from src.app.services.model_type_utils import create_model_type, get_all_model_types, get_model_type_by_name
from src.app.schemas import ModelTypeCreate
from src.app.exceptions import ForeignKeyViolationException, NotFoundException
from src.app.database_tables import ModelTypeTable
import uuid


@pytest.mark.asyncio
async def test_create_model_type_success():
    """Test creating a model type successfully"""
    # Arrange
    mock_db = AsyncMock()
    model_type_data = ModelTypeCreate(name="GraphSAGE")

    # Mock the created object
    mock_model_type = ModelTypeTable(id=uuid.uuid4(), name="GraphSAGE")
    mock_db.refresh = AsyncMock(side_effect=lambda obj: setattr(obj, 'id', mock_model_type.id))

    # Act
    result = await create_model_type(model_type_data, mock_db)

    # Assert
    assert result.name == "GraphSAGE"
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()
    mock_db.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_model_type_integrity_error():
    """Test creating a model type with duplicate name raises exception"""
    # Arrange
    mock_db = AsyncMock()
    mock_db.commit.side_effect = IntegrityError("statement", "params", "orig")
    model_type_data = ModelTypeCreate(name="GraphSAGE")

    # Act & Assert
    with pytest.raises(ForeignKeyViolationException):
        await create_model_type(model_type_data, mock_db)

    mock_db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_all_model_types_empty():
    """Test getting all model types when database is empty returns empty list"""
    # Arrange
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    # Act
    result = await get_all_model_types(mock_db)

    # Assert
    assert result == []

@pytest.mark.asyncio
async def test_get_all_model_types_with_data():
    """Test getting all model types when database has data"""
    # Arrange
    mock_db = AsyncMock()
    mock_types = [
        ModelTypeTable(id=uuid.uuid4(), name="GraphSAGE"),
        ModelTypeTable(id=uuid.uuid4(), name="GCN")
    ]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_types
    mock_db.execute.return_value = mock_result

    # Act
    result = await get_all_model_types(mock_db)

    # Assert
    assert len(result) == 2
    assert result[0].name == "GraphSAGE"
    assert result[1].name == "GCN"


@pytest.mark.asyncio
async def test_get_model_type_by_name_found():
    """Test getting a model type by name when it exists"""
    # Arrange
    mock_db = AsyncMock()
    model_type = ModelTypeTable(id=uuid.uuid4(), name="GraphSAGE")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = model_type
    mock_db.execute.return_value = mock_result

    # Act
    result = await get_model_type_by_name("GraphSAGE", mock_db)

    # Assert
    assert result is not None
    assert result.name == "GraphSAGE"
    mock_db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_model_type_by_name_not_found():
    """Test getting a model type by name when it doesn't exist"""
    # Arrange
    mock_db = AsyncMock()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    # Act
    result = await get_model_type_by_name("NonExistent", mock_db)

    # Assert
    assert result is None
    mock_db.execute.assert_awaited_once()

    # Verify the query was constructed correctly with the name filter
    call_args = mock_db.execute.call_args[0][0]
    assert str(call_args).lower().find("where") != -1  # Verify WHERE clause exists
    assert "NonExistent" in str(call_args.compile()) or "name" in str(call_args)
