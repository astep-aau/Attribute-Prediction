import pytest
import torch
import os
import pandas as pd
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.static_graph_builder import StaticGraphBuilder
from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.data_manipulation.data_pipeline import setup_data_loaders
import json


def test_full_pipeline_integrity(tmp_path):
    """
    Simulates a full run from CSV to DataLoader.
    """
    # 1. SETUP: Create tiny mock CSVs and JSON in a temporary directory
    d = tmp_path / "data"
    d.mkdir()
    traffic_file = d / "day1.csv"
    conn_file = d / "conn.csv"
    meta_file = d / "meta.json"  # Define a path for the meta file

    pd.DataFrame({
        "time_slot": ["00:00", "00:05"],
        "edge0_traversal_time_sec": [10, 12],
        "edge1_traversal_time_sec": [20, 22],
        "edge2_traversal_time_sec": [30, 32]
    }).to_csv(traffic_file, index=False)

    # 2. Connections: Edge IDs should be 0, 1, 2 to match the columns
    pd.DataFrame({
        "edge_id": [0, 1, 2],
        "vertex_start_id": [100, 101, 102],
        "vertex_end_id": [101, 102, 103]
    }).to_csv(conn_file, index=False)

    # 3. Metadata: Keys must match the integer IDs
    mock_meta = {
        "1": {
            "nodes": [100, 101],
            "road_type": "primary",
            "oneway": True
        },
        "2": {
            "nodes": [101, 102],
            "road_type": "secondary",
            "oneway": False
        },
        "3": {
            "nodes": [102, 103],
            "road_type": "tertiary",
            "oneway": False
        }
    }
    with open(meta_file, 'w') as f:
        json.dump(mock_meta, f)

    # 4. EXECUTION: Match everything up
    master_cols = [
        "time_slot",
        "edge0_traversal_time_sec",
        "edge1_traversal_time_sec",
        "edge2_traversal_time_sec"
    ]
    edge_ids = [0, 1, 2]

    loader = FileLoader(str(traffic_file), str(conn_file), str(meta_file), master_cols)
    static_builder = StaticGraphBuilder(loader, edge_ids)
    static_components = static_builder.get_static_components()

    builder = GraphDatasetBuilder(loader, day_of_week_index=0, **static_components)

    # Pipeline
    seq_len = 1
    train, val, test = setup_data_loaders([builder], seq_len=seq_len, mask_rate=0.5, batch_size=1)

    # 3. ASSERTIONS
    batch = next(iter(train))

    # Check shape: (B, N, T, F) -> (1, 3, 1, F)
    assert batch['x_combined'].shape[0] == 1  # Batch
    assert batch['x_combined'].shape[1] == 3  # Nodes
    assert batch['x_combined'].shape[2] == seq_len  # Time

    # Verify values: Does edge3 speed still equal 30?
    # This ensures no columns were swapped during builder initialization
    edge3_val = batch['y_true'][0, 2, 0].item()
    assert edge3_val in [30.0, 32.0]