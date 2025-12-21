import pytest
import torch
from src.data_manipulation.data_pipeline import SequenceDataset


@pytest.fixture
def mock_dataset_params():
    """Sets up dimensions and tensors for a small mock dataset."""
    num_nodes = 5
    total_timesteps = 50
    seq_len = 12

    # Traffic matrix with some 'holes' (-1.0)
    data_matrix = torch.rand((total_timesteps, num_nodes)) * 100
    data_matrix[5, 0] = -1.0  # Inject a hole at T=5, Node=0

    temporal_features = torch.randn((total_timesteps, 9))
    static_features = torch.randn((num_nodes, 22))
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_ids = torch.arange(num_nodes)

    return {
        "data_matrix": data_matrix,
        "temporal_features": temporal_features,
        "static_features": static_features,
        "edge_index": edge_index,
        "edge_ids": edge_ids,
        "seq_len": seq_len
    }


def test_dataset_length(mock_dataset_params):
    """Verify length is Total_T - Seq_Len + 1."""
    ds = SequenceDataset(**mock_dataset_params, mask_rate=0.0)
    expected_len = 50 - 12 + 1
    assert len(ds) == expected_len


def test_output_shapes(mock_dataset_params):
    """Check if the combined feature tensor has the correct (N, T, F) shape."""
    ds = SequenceDataset(**mock_dataset_params, mask_rate=0.2)
    sample = ds[0]

    # x_combined: (N=5, T=12, F=1 + 9 + 22 = 32)
    assert sample['x_combined'].shape == (5, 12, 32)
    # y_true and mask: (N=5, T=12)
    assert sample['y_true'].shape == (5, 12)
    assert sample['mask'].shape == (5, 12)


def test_masking_logic_training(mock_dataset_params):
    """Ensure that with mask_rate > 0, x_combined is zeroed and mask is generated."""
    # Set mask_rate to 1.0 (mask everything) to make verification easy
    ds = SequenceDataset(**mock_dataset_params, mask_rate=1.0)
    sample = ds[0]

    # The first feature in x_combined is the traffic traversal time
    traffic_input = sample['x_combined'][:, :, 0]

    # Since we masked 100%, all inputs should be 0.0
    assert torch.all(traffic_input == 0.0)
    # The mask should identify where we need to predict (all points that weren't original holes)
    assert sample['mask'].any()


def test_masking_logic_inference(mock_dataset_params):
    """Ensure that with mask_rate = 0, only original holes are masked."""
    ds = SequenceDataset(**mock_dataset_params, mask_rate=0.0)

    # We know we put a hole at T=5, Node=0. In index 0, this is valid.
    sample = ds[0]

    # Check if the sentinel -1.0 was converted to 0.0 in the input
    # Node 0 is index 0, Time 5 is index 5
    assert sample['x_combined'][0, 5, 0] == 0.0

    # Check if that hole is correctly identified in the mask
    assert sample['mask'][0, 5] == True

    # Check that a 'good' point (e.g., Node 1, Time 1) is NOT masked
    assert sample['mask'][1, 1] == False


def test_temporal_feature_repetition(mock_dataset_params):
    """Verify that temporal features are repeated correctly across all nodes."""
    ds = SequenceDataset(**mock_dataset_params, mask_rate=0.0)
    sample = ds[0]

    # Temporal features start at index 1 and end at index 9 (total 10) in the last dim
    node0_temp = sample['x_combined'][0, :, 1:10]
    node1_temp = sample['x_combined'][1, :, 1:10]

    # All nodes should see the exact same temporal features for the same time window
    assert torch.equal(node0_temp, node1_temp)