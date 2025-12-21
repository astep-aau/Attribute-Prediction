import pytest
import torch
import pandas as pd
from unittest.mock import MagicMock
from src.data_manipulation.static_graph_builder import StaticGraphBuilder


@pytest.fixture
def mock_loader():
    """Provides a mocked FileLoader with controlled return values."""
    loader = MagicMock()

    # 1. Mock Adjacency: Two edges that connect at vertex 2
    # Edge 10: 1 -> 2 | Edge 20: 2 -> 3
    adj_df = pd.DataFrame({
        "edge_id": [10, 20],
        "vertex_start_id": [1, 2],
        "vertex_end_id": [2, 3]
    })

    # 2. Mock Metadata: Way 999 contains vertex 1 and 2 (matches Edge 10)
    # way index must be int as per FileLoader.get_meta_data()
    meta_df = pd.DataFrame({
        "nodes": [["1", "2"], ["2", "3"]],
        "road_type": ["motorway", "residential"],
        "oneway": [True, False]
    }, index=[999, 888])

    loader.get_adjacency.return_value = adj_df
    loader.get_meta_data.return_value = meta_df
    return loader


def test_create_edge_as_node_map(mock_loader):
    """Verify physical IDs are mapped to sequential 0-N indices."""
    global_ids = [20, 10]  # Unordered input
    builder = StaticGraphBuilder(mock_loader, global_ids)

    # Mapping should be sorted: 10 -> 0, 20 -> 1
    assert builder.edge_as_node_map[10] == 0
    assert builder.edge_as_node_map[20] == 1
    assert torch.equal(builder.edge_ids, torch.tensor([10, 20], dtype=torch.int))


def test_build_node_features_encoding(mock_loader):
    """Test that road types and oneway status are correctly encoded into tensors."""
    global_ids = [10, 20]
    builder = StaticGraphBuilder(mock_loader, global_ids)

    # Road types index 0 is 'motorway'.
    # Edge 10 is motorway + oneway.
    # Expected: [1, 0, 0, ..., 1] (oneway is the last index)
    features = builder.x
    assert features.shape == (2, 22)  # 21 road types + 1 oneway

    # Edge 10 (Node 0)
    assert features[0, 0] == 1.0  # motorway bit
    assert features[0, -1] == 1.0  # oneway bit

    # Edge 20 (Node 1) is 'residential' (index 6 in road_types list)
    assert features[1, 6] == 1.0
    assert features[1, -1] == 0.0  # not oneway


def test_line_graph_edge_index(mock_loader):
    """Verify that Edge 10 -> Edge 20 connection is captured."""
    global_ids = [10, 20]
    builder = StaticGraphBuilder(mock_loader, global_ids)

    # Edge 10 (Node 0) ends at Vertex 2.
    # Edge 20 (Node 1) starts at Vertex 2.
    # Connection: 0 -> 1
    expected_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    assert torch.equal(builder.edge_index, expected_edge_index)


def test_missing_metadata_defaults_to_zeros(mock_loader):
    """Verify that edges with no matching metadata get zeroed features."""
    # Provide an edge ID (30) that has no vertices in the metadata nodes list
    global_ids = [30]
    builder = StaticGraphBuilder(mock_loader, global_ids)

    # Should result in a vector of all zeros
    assert torch.all(builder.x == 0)


def test_get_static_components_keys(mock_loader):
    """Ensure the returned dictionary has the expected keys for the training pipeline."""
    builder = StaticGraphBuilder(mock_loader, [10, 20])
    components = builder.get_static_components()

    expected_keys = {'x', 'edge_index', 'edge_ids', 'edge_as_node_map'}
    assert set(components.keys()) == expected_keys