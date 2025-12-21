import pytest
import torch
import torch.nn as nn
from src.models.GAT_BiGRU import GAT_BiGRU_Imputer
from src.models.trainer import train_epoch


def test_model_convergence_on_small_data():
    """
    Integration test: Can the GAT-BiGRU model overfit a single batch?
    This proves that the Model + Loss + Optimizer + Device handoff is working.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Setup Model
    model = GAT_BiGRU_Imputer(
        in_feat=16, gnn_hidden=32, gru_hidden=32, num_gnn_layers=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss(reduction='none')

    # 2. Create a stable single batch
    B, T, N, F = 1, 4, 5, 16
    x = torch.randn((B, N, T, F)).to(device)  # Pipeline format
    y_true = torch.randn((B, N, T)).to(device)
    mask = torch.ones_like(y_true).bool()
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long).to(device)

    batch = {'x_combined': x, 'y_true': y_true, 'mask': mask, 'edge_index': edge_index.unsqueeze(0)}
    loader = [batch]  # Mock loader

    # 3. Training loop
    initial_loss = None
    for epoch in range(50):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        if initial_loss is None: initial_loss = loss

    # 4. ASSERT: Loss must have significantly decreased
    assert loss < (initial_loss * 0.1), f"Model failed to converge. Final loss {loss} is too high."


def test_checkpoint_consistency(tmp_path):
    """
    Integration test: Does saving/loading produce identical predictions?
    """
    model = GAT_BiGRU_Imputer(16, 32, 32)
    model.eval()

    # Mock input
    x = torch.randn((1, 4, 5, 16))
    edge_index = torch.zeros((1, 2, 0), dtype=torch.long)

    # Save
    save_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), save_path)

    # Load into NEW model instance
    new_model = GAT_BiGRU_Imputer(16, 32, 32)
    new_model.load_state_dict(torch.load(save_path))
    new_model.eval()

    # Compare
    with torch.no_grad():
        out1 = model(x.permute(0, 2, 1, 3), edge_index)
        out2 = new_model(x.permute(0, 2, 1, 3), edge_index)

    assert torch.allclose(out1, out2), "Checkpoint loading altered model predictions."