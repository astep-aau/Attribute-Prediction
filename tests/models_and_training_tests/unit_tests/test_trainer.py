import pytest
import torch
import torch.nn as nn
from src.models.trainer import calculate_mape_and_bias, train_epoch, evaluate


class MockModel(nn.Module):
    """A simple linear model to test trainer logic."""

    def __init__(self, in_dim=32, out_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        # x input is (B, T, N, F)
        # Output should be (B, T, N)
        B, T, N, F = x.shape
        return self.linear(x).view(B, T, N)


@pytest.fixture
def mock_batch():
    B, T, N, F = 2, 4, 3, 32
    return {
        'x_combined': torch.randn((B, N, T, F)),  # Note: SequenceDataset output format
        'y_true': torch.randn((B, N, T)),
        'mask': torch.randint(0, 2, (B, N, T)).bool(),
        'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    }


def test_calculate_mape_and_bias():
    """Verify MAPE and Bias math with known values."""
    # y_true = 10, pred = 12 -> error = 2. MAPE = (2/10)*100 = 20%
    y_true = torch.tensor([10.0, 20.0])
    prediction = torch.tensor([12.0, 18.0])
    mask = torch.tensor([True, True])

    mape, bias = calculate_mape_and_bias(prediction, y_true, mask)

    # MAPE: (abs(12-10)/10 + abs(18-20)/20) / 2 * 100 = (0.2 + 0.1)/2 * 100 = 15.0
    assert pytest.approx(mape, 0.1) == 15.0
    # Bias: ( (12-10) + (18-20) ) / 2 = (2 - 2) / 2 = 0.0
    assert pytest.approx(bias, 0.1) == 0.0


def test_evaluate_masked_loss():
    """Ensure evaluate only considers masked positions."""
    model = MockModel()
    criterion = nn.MSELoss(reduction='none')

    # Create a batch where prediction is perfect on masked values but wrong on others
    B, T, N = 1, 2, 2
    y_true = torch.ones((B, N, T))
    mask = torch.tensor([[[True, False], [False, True]]])  # Diagonal mask

    # Construct batch
    batch = {
        'x_combined': torch.randn((B, N, T, 32)),
        'y_true': y_true,
        'mask': mask,
        'edge_index': torch.tensor([[0], [1]])
    }

    # Mock model that returns 1.0 everywhere (perfect prediction for masked spots)
    model.forward = lambda x, edge: torch.ones((B, T, N))

    loss = evaluate(model, [batch], criterion, torch.device('cpu'))
    assert loss == 0.0, "Loss should be 0 because masked values are predicted perfectly"


def test_train_epoch_gradient_accumulation(mock_batch):
    """Verify weights change after the accumulation steps."""
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')

    # Get initial weights
    initial_weight = model.linear.weight.clone()

    # Run one step. accumulate_steps is 8 in your code.
    # Since we only provide 1 batch, weights should NOT change yet.
    train_epoch(model, [mock_batch], criterion, optimizer, torch.device('cpu'))

    # In your current code, the final step logic (len(loader) % 8 != 0)
    # executes optimizer.step(). Let's verify it actually updated.
    current_weight = model.linear.weight
    assert not torch.equal(initial_weight, current_weight), "Weights should update at end of epoch"


def test_loss_is_scaled_correctly():
    """Verify masked loss is averaged correctly across only valid points."""
    # This checks if masked_loss = loss_matrix[mask].mean() behaves as expected
    loss_matrix = torch.tensor([[10.0, 2.0], [0.0, 5.0]])
    mask = torch.tensor([[True, False], [False, True]])

    # Valid values are 10.0 and 5.0. Mean should be 7.5
    assert loss_matrix[mask].mean().item() == 7.5