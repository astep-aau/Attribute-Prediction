import pytest
import torch
from unittest.mock import MagicMock
from src.data_manipulation.data_pipeline import setup_data_loaders


@pytest.fixture
def mock_builders():
    """Creates a list of 5 mock builders to represent Day 3 through Day 7."""
    builders = []
    num_nodes = 5
    timesteps_per_day = 100

    for i in range(5):
        builder = MagicMock()
        # Mock traffic matrix (T, N)
        builder.get_full_traffic_matrix.return_value = torch.rand((timesteps_per_day, num_nodes))
        # Mock temporal features (T, 9)
        builder.temporal_features = torch.randn((timesteps_per_day, 9))
        # Mock static features
        builder.x = torch.randn((num_nodes, 22))
        builder.edge_index = torch.tensor([[0], [1]])
        builder.edge_ids = torch.tensor([1, 2, 3, 4, 5])
        builders.append(builder)
    return builders


def test_aggregation_shapes(mock_builders):
    """Verify that 5 builders of 100 steps each result in a 500-step aggregate."""
    seq_len = 10
    # We only need to check if it runs without crashing and basic stats
    train_l, val_l, test_l = setup_data_loaders(mock_builders, seq_len, 0.2, 1)

    # Each loader's dataset should point to the same aggregated matrix
    # Total steps = 5 * 100 = 500
    assert len(train_l.dataset.traffic_matrix) == 500


def test_split_logic_sample_counts(mock_builders):
    """Verify the number of samples in Train, Val, and Test matches the math."""
    seq_len = 10
    T_day = 100
    # Train uses 2 days (index 0, 2) with step_size=1
    # Samples per day = T_day - seq_len + 1 = 100 - 10 + 1 = 91
    # Total Train = 91 * 2 = 182

    # Val uses 2 days (index 1, 3) with step_size=seq_len (10)
    # Samples per day = floor((100 - 10) / 10) + 1 = 10
    # Total Val = 10 * 2 = 20

    train_l, val_l, test_l = setup_data_loaders(mock_builders, seq_len, 0.2, 1)

    assert len(train_l.dataset) == 182
    assert len(val_l.dataset) == 20
    assert len(test_l.dataset) == 10  # 1 day, step 10


def test_no_leakage(mock_builders):
    """Ensure that Test indices do not overlap with Train indices."""
    seq_len = 10
    train_l, val_l, test_l = setup_data_loaders(mock_builders, seq_len, 0.2, 1)

    train_indices = set(train_l.dataset.active_indices)
    test_indices = set(test_l.dataset.active_indices)

    # The intersection of these two sets must be empty
    intersection = train_indices.intersection(test_indices)
    assert len(intersection) == 0


def test_empty_builder_list():
    """Ensure the function raises an error if no builders are provided."""
    with pytest.raises(ValueError, match="builders_list cannot be empty."):
        setup_data_loaders([], 12, 0.2, 1)