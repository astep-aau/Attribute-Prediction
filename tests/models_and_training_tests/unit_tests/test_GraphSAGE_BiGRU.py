import pytest
import torch
from src.models.GraphSAGE_BiGRU import GraphSAGE_BiGRU_Imputer


@pytest.fixture
def sage_params():
    return {
        "in_feat": 32,
        "gnn_hidden": 64,
        "gru_hidden": 128,
        "num_gnn_layers": 1,
        "dropout": 0.2
    }


def test_sage_output_dimensions(sage_params):
    """Verify final output tensor is (B, T, N)."""
    model = GraphSAGE_BiGRU_Imputer(**sage_params)
    B, T, N, F = 4, 12, 20, 32

    x = torch.randn((B, T, N, F))
    # Mock edges: Node 0 connected to Node 1 in each batch
    edge_index = torch.tensor([[0], [1]], dtype=torch.long).unsqueeze(0).repeat(B, 1, 1)

    output = model(x, edge_index)

    assert output.shape == (B, T, N)
    assert not torch.isnan(output).any()


def test_sage_multi_layer_flow(sage_params):
    """Verify that stacking multiple GraphSAGE layers doesn't break feature dimensions."""
    params = sage_params.copy()
    params["num_gnn_layers"] = 3
    model = GraphSAGE_BiGRU_Imputer(**params)

    B, T, N, F = 1, 5, 10, 32
    x = torch.randn((B, T, N, F))
    edge_index = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)

    # If the internal in_dim update in __init__ is wrong, this will crash
    output = model(x, edge_index)
    assert output.shape == (1, 5, 10)


def test_spatial_influence(sage_params):
    """Ensure GraphSAGE is actually aggregating information from neighbors."""
    model = GraphSAGE_BiGRU_Imputer(**sage_params)
    model.eval()  # Disable dropout for deterministic check

    B, T, N, F = 1, 1, 3, 32
    x = torch.ones((B, T, N, F))

    # Case 1: No edges (Isolated nodes)
    edge_index_none = torch.zeros((B, 2, 0), dtype=torch.long)

    # Case 2: Node 0 is connected to Node 1 and 2
    edge_index_connected = torch.tensor([[[1, 2], [0, 0]]], dtype=torch.long)

    with torch.no_grad():
        out_isolated = model(x, edge_index_none)
        out_connected = model(x, edge_index_connected)

    # The value for Node 0 should change because its features now include neighbor info
    assert not torch.allclose(out_isolated[0, 0, 0], out_connected[0, 0, 0]), \
        "GraphSAGE failed to aggregate spatial information from neighbors."


def test_batch_independence(sage_params):
    """Ensure that data from Batch 0 does not leak into Batch 1."""
    model = GraphSAGE_BiGRU_Imputer(**sage_params)
    B, T, N, F = 2, 5, 5, 32

    x = torch.randn((B, T, N, F))
    edge_index = torch.zeros((B, 2, 0), dtype=torch.long)

    # Forward pass on two batches
    out_both = model(x, edge_index)

    # Forward pass on just the first batch
    out_single = model(x[0:1], edge_index[0:1])

    # Results for Batch 0 should be identical
    assert torch.allclose(out_both[0], out_single[0], atol=1e-5), \
        "Batch leakage detected: Batch 1 influenced Batch 0 results."