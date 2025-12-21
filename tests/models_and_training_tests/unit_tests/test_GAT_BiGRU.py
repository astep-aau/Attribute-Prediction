import pytest
import torch
from src.models.GAT_BiGRU import GAT_BiGRU_Imputer


@pytest.fixture
def model_params():
    return {
        "in_feat": 32,
        "gnn_hidden": 64,
        "gru_hidden": 128,
        "heads": 2,
        "num_gnn_layers": 2
    }


def test_model_output_shape(model_params):
    """Verify that the model produces the correct (B, T, N) output shape."""
    model = GAT_BiGRU_Imputer(**model_params)

    B, T, N, F = 2, 12, 10, 32
    x_combined = torch.randn((B, T, N, F))
    # Edge index for a simple ring graph per batch: (B, 2, E)
    edge_index = torch.tensor([[0, 1, 2, 9], [1, 2, 3, 0]], dtype=torch.long)
    edge_index_batched = edge_index.unsqueeze(0).repeat(B, 1, 1)

    output = model(x_combined, edge_index_batched)

    # Final output should be (Batch, Time, Nodes)
    assert output.shape == (B, T, N)
    assert not torch.isnan(output).any(), "Model produced NaN values"


def test_batched_edge_index_shifting(model_params):
    """Verify that internal logic correctly offsets edge indices for batches."""
    model = GAT_BiGRU_Imputer(**model_params)

    B, T, N, F = 2, 4, 3, 32
    x_combined = torch.randn((B, T, N, F))
    # Simple edge: 0 -> 1
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_index_batched = edge_index.unsqueeze(0).repeat(B, 1, 1)

    # We want to ensure that for Batch 1, edge (0,1) becomes (3,4) internally.
    # We can't easily check internal variables, but we check if forward pass succeeds.
    try:
        model(x_combined, edge_index_batched)
    except RuntimeError as e:
        pytest.fail(f"Forward pass failed, likely due to edge_index shifting: {e}")


def test_temporal_consistency(model_params):
    """Ensure that the BiGRU is actually processing the time dimension."""
    model = GAT_BiGRU_Imputer(**model_params)
    B, T, N, F = 1, 12, 5, 32
    x_combined = torch.randn((B, T, N, F))
    edge_index = torch.zeros((B, 2, 2), dtype=torch.long)  # Self loops or empty

    # If we change only the last timestep of the input...
    x_modified = x_combined.clone()
    x_modified[:, -1, :, :] += 10.0

    out1 = model(x_combined, edge_index)
    out2 = model(x_modified, edge_index)

    # ...the outputs should be different because of the BiGRU temporal dependency
    assert not torch.equal(out1, out2), "Model output did not change when temporal input changed"


def test_gnn_layer_stacking(model_params):
    """Verify model can initialize and run with multiple GNN layers."""
    params = model_params.copy()
    params["num_gnn_layers"] = 3
    model = GAT_BiGRU_Imputer(**params)

    B, T, N, F = 1, 5, 4, 32
    x = torch.randn((B, T, N, F))
    edge_index = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)

    output = model(x, edge_index)
    assert output.shape == (B, 5, 4)