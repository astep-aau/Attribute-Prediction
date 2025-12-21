import pytest
import pandas as pd
import json
import os
from src.data_manipulation.file_loader import FileLoader


@pytest.fixture
def mock_files(tmp_path):
    """Fixture to create temporary files for testing."""
    d = tmp_path / "data"
    d.mkdir()

    # 1. Create a sample travel data CSV (missing one edge column)
    travel_data_path = d / "travel_data.csv"
    travel_df = pd.DataFrame({
        "time_slot": ["00:00", "01:30"],
        "edge1_sec": [10.0, 20.0]
    })
    travel_df.to_csv(travel_data_path, index=False)

    # 2. Create a sample adjacency CSV
    adj_path = d / "adj.csv"
    adj_df = pd.DataFrame({"from": [1], "to": [2]})
    adj_df.to_csv(adj_path, index=False)

    # 3. Create a sample metadata JSON
    meta_path = d / "meta.json"
    meta_content = {
        "1": {"length": 100, "highway": "primary"},
        "2": {"length": 50, "highway": "secondary"}
    }
    with open(meta_path, 'w') as f:
        json.dump(meta_content, f)

    return str(travel_data_path), str(adj_path), str(meta_path)


def test_get_travel_data_padding_and_order(mock_files):
    """Test that missing columns are padded with -1.0 and order is preserved."""
    travel_path, adj_path, meta_path = mock_files
    # master_cols includes 'edge2_sec' which is NOT in the mock CSV
    master_cols = ["time_slot", "edge1_sec", "edge2_sec"]

    loader = FileLoader(travel_path, adj_path, meta_path, master_cols)
    df = loader.get_travel_data()

    # Check if edge2_sec was created and filled with -1.0
    assert "edge2_sec" in df.columns
    assert (df["edge2_sec"] == -1.0).all()

    # Check if the column order matches master_cols exactly
    # Note: Timestamp is added by the function, so we check first N cols
    assert list(df.columns[:3]) == master_cols


def test_timestamp_calculation(mock_files):
    """Test that time_slot strings are correctly converted to Datetime objects."""
    travel_path, adj_path, meta_path = mock_files
    master_cols = ["time_slot", "edge1_sec"]

    loader = FileLoader(travel_path, adj_path, meta_path, master_cols)
    df = loader.get_travel_data()

    # Check specific timestamp values
    # Base date is 2025-11-26
    assert df["Timestamp"].iloc[0] == pd.Timestamp("2025-11-26 00:00:00")
    assert df["Timestamp"].iloc[1] == pd.Timestamp("2025-11-26 01:30:00")


def test_get_meta_data_structure(mock_files):
    """Test that metadata is transposed and index is integer."""
    travel_path, adj_path, meta_path = mock_files
    loader = FileLoader(travel_path, adj_path, meta_path, ["time_slot"])

    meta_df = loader.get_meta_data()

    # Check transposition (original keys "1", "2" should be the index now)
    assert 1 in meta_df.index
    assert 2 in meta_df.index
    assert meta_df.index.dtype == "int64"
    assert "length" in meta_df.columns


def test_get_adjacency(mock_files):
    """Test basic loading of connectivity data."""
    travel_path, adj_path, meta_path = mock_files
    loader = FileLoader(travel_path, adj_path, meta_path, ["time_slot"])

    adj_df = loader.get_adjacency()
    assert not adj_df.empty
    assert "from" in adj_df.columns