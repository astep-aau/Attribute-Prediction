from uuid import UUID
from pathlib import Path
import logging
import pandas as pd
import aiofiles
import asyncio
from typing import Optional
from src.app.schemas import (
    ImputeResultResponse,
    RoadIdResponse,
    TimeIntervalResponse,
    ImputeResultCreate,
)
from src.app.exceptions import NotFoundException, InvalidUUIDException
from src.app.config import settings

logger = logging.getLogger(__name__)

# Lock for thread-safe CSV operations
_csv_locks = {}

def _get_csv_lock(model_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific model's CSV file"""
    if model_id not in _csv_locks:
        _csv_locks[model_id] = asyncio.Lock()
    return _csv_locks[model_id]

def _get_csv_path(model_id: str) -> Path:
    """Get the CSV file path for a model"""
    csv_dir = settings.impute_results_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir / f"{model_id}.csv"

async def _read_csv(model_id: str) -> Optional[pd.DataFrame]:
    """Read CSV file for a model, return None if file doesn't exist"""
    csv_path = _get_csv_path(model_id)

    if not csv_path.exists():
        return None

    try:
        # Read CSV in executor to avoid blocking
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: pd.read_csv(csv_path, dtype={'road_id': str, 'tms': int, 'value': float, 'imputed': bool})
        )
        return df
    except Exception as e:
        logger.error(f"Error reading CSV for model {model_id}: {str(e)}")
        raise

async def _write_csv(model_id: str, df: pd.DataFrame):
    """Write DataFrame to CSV file"""
    csv_path = _get_csv_path(model_id)

    try:
        # Write CSV in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: df.to_csv(csv_path, index=False)
        )
    except Exception as e:
        logger.error(f"Error writing CSV for model {model_id}: {str(e)}")
        raise

async def find_impute_results(
        model_id: str,
        road_id: str,
        start_time: int,
        end_time: int):
    """
    Find imputation results for a specific model and road within a time range

    Args:
        model_id: UUID string of the model
        road_id: ID of the road
        start_time: Start time as Unix timestamp
        end_time: End time as Unix timestamp

    Returns:
        List of ImputeResultResponse objects with timestamps, values, and imputed flags

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no results found for the given parameters
    """
    try:
        UUID(model_id)
    except ValueError:
        logger.warning(f"Invalid UUID format for model_id: {model_id}")
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")

    logger.debug(f"Reading impute results for model: {model_id}, road: {road_id}, time range: {start_time}-{end_time}")

    df = await _read_csv(model_id)

    if df is None or df.empty:
        logger.info(f"No impute results found for model {model_id}")
        raise NotFoundException(f"No impute results found for model {model_id}")

    # Filter by road_id and time range
    filtered_df = df[
        (df['road_id'] == road_id) &
        (df['tms'] >= start_time) &
        (df['tms'] <= end_time)
    ]

    if filtered_df.empty:
        logger.info(f"No impute results found for model {model_id}, road {road_id}")
        raise NotFoundException(f"No impute results found for model {model_id}, road {road_id}")

    logger.info(f"Found {len(filtered_df)} impute results for model {model_id}, road {road_id}")

    # Convert to response objects
    response = [
        ImputeResultResponse(
            tms=int(row['tms']),
            value=float(row['value']),
            imputed=bool(row['imputed'])
        )
        for _, row in filtered_df.iterrows()
    ]

    return response

async def find_road_ids(model_id: str):
    """
    Find all distinct road IDs that have imputation data for a model

    Args:
        model_id: UUID string of the model

    Returns:
        List of RoadIdResponse objects containing unique road IDs

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no roads found for the model
    """
    try:
        UUID(model_id)
    except ValueError:
        logger.warning(f"Invalid UUID format for model_id: {model_id}")
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")

    logger.debug(f"Reading road IDs for model: {model_id}")

    df = await _read_csv(model_id)

    if df is None or df.empty:
        logger.info(f"No roads found for model {model_id}")
        raise NotFoundException(f"No roads found for model {model_id}")

    # Get unique road IDs
    unique_roads = df['road_id'].unique().tolist()

    if not unique_roads:
        logger.info(f"No roads found for model {model_id}")
        raise NotFoundException(f"No roads found for model {model_id}")

    logger.info(f"Found {len(unique_roads)} roads for model {model_id}")
    return [RoadIdResponse(road_id=str(road_id)) for road_id in unique_roads]

async def find_timespan(model_id: str, road_id: str):
    """
    Find the minimum and maximum timestamps for imputation data

    Args:
        model_id: UUID string of the model
        road_id: ID of the road

    Returns:
        TimeIntervalResponse with start_time and end_time as Unix timestamps

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
        NotFoundException: If no timestamps found for the given model and road
    """
    try:
        UUID(model_id)
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {model_id}")

    df = await _read_csv(model_id)

    if df is None or df.empty:
        raise NotFoundException(f"No data found for model {model_id}")

    # Filter by road_id
    filtered_df = df[df['road_id'] == road_id]

    if filtered_df.empty:
        raise NotFoundException(f"No data found for model {model_id}, road {road_id}")

    min_time = int(filtered_df['tms'].min())
    max_time = int(filtered_df['tms'].max())

    return TimeIntervalResponse(start_time=min_time, end_time=max_time)

async def create_impute_result(result_data: ImputeResultCreate):
    """
    Create a new imputation result entry by appending to CSV

    Args:
        result_data: Imputation result data including model_id, road_id, tms, value, and imputed

    Returns:
        The created ImputeResultResponse object

    Raises:
        InvalidUUIDException: If model_id is not a valid UUID
    """
    try:
        UUID(str(result_data.model_id))
    except ValueError:
        raise InvalidUUIDException(f"Invalid UUID format: {result_data.model_id}")

    logger.info(f"Creating impute result for model: {result_data.model_id}, road: {result_data.road_id}, timestamp: {result_data.tms}")

    model_id_str = str(result_data.model_id)
    lock = _get_csv_lock(model_id_str)

    async with lock:
        # Read existing data
        df = await _read_csv(model_id_str)

        # Create new row
        new_row = pd.DataFrame([{
            'road_id': result_data.road_id,
            'tms': result_data.tms,
            'value': result_data.value,
            'imputed': result_data.imputed
        }])

        if df is None:
            # Create new CSV
            df = new_row
        else:
            # Check for duplicate (same road_id and tms)
            duplicate = df[(df['road_id'] == result_data.road_id) & (df['tms'] == result_data.tms)]
            if not duplicate.empty:
                logger.warning(f"Duplicate entry found for road {result_data.road_id}, timestamp {result_data.tms}")
                # Update existing row instead of adding duplicate
                df.loc[(df['road_id'] == result_data.road_id) & (df['tms'] == result_data.tms),
                       ['value', 'imputed']] = [result_data.value, result_data.imputed]
            else:
                # Append new row
                df = pd.concat([df, new_row], ignore_index=True)

        # Write back to CSV
        await _write_csv(model_id_str, df)

    logger.info(f"Impute result created successfully for road: {result_data.road_id}")

    return ImputeResultResponse(
        tms=result_data.tms,
        value=result_data.value,
        imputed=result_data.imputed
    )
