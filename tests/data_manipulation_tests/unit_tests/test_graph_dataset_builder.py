import pytest
import torch
import pandas as pd
import numpy as np
from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder


@pytest.fixture
def mock_static_components():
    """Provides pre-built static graph components."""
    # 2 nodes, 2 static features each
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_ids = torch.tensor([10, 20], dtype=torch.int)
    edge_as_node_map = {10: 0, 20: 1}

    return x, edge_index, edge_ids, edge_as_node_map


@pytest.fixture
def mock_loader_with_data():
    """Provides a mocked FileLoader that returns a sample traffic DataFrame."""

    class MockLoader:
        def get_travel_data(self):
            return pd.DataFrame({
                "Timestamp": pd.to_datetime(["2025-11-26 00:00:00", "2025-11-26 12:00:00"]),
                "edge20_traversal_time_sec": [30.0, 40.0],  # Node 1
                "edge10_traversal_time_sec": [10.0, 15.0],  # Node 0
                "time_slot": ["00:00", "12:00"]
            })

    return MockLoader()


def test_traffic_matrix_alignment(mock_loader_with_data, mock_static_components):
    """Verify that traffic columns are mapped to the correct Node IDs."""
    x, edge_index, edge_ids, edge_map = mock_static_components
    builder = GraphDatasetBuilder(mock_loader_with_data, 2, x, edge_index, edge_ids, edge_map)

    matrix = builder.get_full_traffic_matrix()

    # Shape should be (TimeSteps=2, Nodes=2)
    assert matrix.shape == (2, 2)

    # Column 0 should be edge 10 (10.0, 15.0)
    # Column 1 should be edge 20 (30.0, 40.0)
    assert matrix[0, 0] == 10.0
    assert matrix[1, 0] == 15.0
    assert matrix[0, 1] == 30.0
    assert matrix[1, 1] == 40.0


def test_temporal_feature_dimensions(mock_loader_with_data, mock_static_components):
    """Verify temporal features produce the correct (T, 9) shape."""
    x, edge_index, edge_ids, edge_map = mock_static_components
    # Day index 2 = Wednesday
    builder = GraphDatasetBuilder(mock_loader_with_data, 2, x, edge_index, edge_ids, edge_map)

    temp_feats = builder.temporal_features

    # (2 timesteps, 2 cyclic + 7 one-hot = 9)
    assert temp_feats.shape == (2, 9)

    # Check One-Hot for Day 2 (index 4 in the 9-dim vector: sin, cos, dow0, dow1, DOW2...)
    assert temp_feats[0, 4] == 1.0
    assert temp_feats[0, 2] == 0.0  # dow0 should be 0


def test_cyclic_time_encoding(mock_loader_with_data, mock_static_components):
    """Verify that midnight and noon result in expected sin/cos values."""
    x, edge_index, edge_ids, edge_map = mock_static_components
    builder = GraphDatasetBuilder(mock_loader_with_data, 2, x, edge_index, edge_ids, edge_map)

    temp_feats = builder.temporal_features

    # At 00:00, sin(0) = 0, cos(0) = 1
    assert pytest.approx(temp_feats[0, 0].item(), abs=1e-5) == 0.0
    assert pytest.approx(temp_feats[0, 1].item(), abs=1e-5) == 1.0

    # At 12:00 (midday), sin(pi) = 0, cos(pi) = -1
    assert pytest.approx(temp_feats[1, 0].item(), abs=1e-5) == 0.0
    assert pytest.approx(temp_feats[1, 1].item(), abs=1e-5) == -1.0


def test_get_data_pyg(mock_loader_with_data, mock_static_components):
    """Ensure it returns a valid PyTorch Geometric Data object."""
    x, edge_index, edge_ids, edge_map = mock_static_components
    builder = GraphDatasetBuilder(mock_loader_with_data, 2, x, edge_index, edge_ids, edge_map)

    data = builder.get_data()
    assert data.x.shape == (2, 2)
    assert data.edge_index.shape == (2, 1)