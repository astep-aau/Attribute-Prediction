"""
Full training script for GraphSAGE-GRU model on road network data.
"""

from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.temporal_dataset_builder import TemporalDatasetBuilder
from src.models.graphsage_gru import GraphSAGEGru
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time

# ===== CONFIGURATION =====
CONFIG = {
    # Data
    'sequence_length': 5,
    'batch_size': 1,  # Start with 1, model processes full graph
    'train_split': 0.8,

    # Model
    'gnn_hidden_dim': 64,
    'gru_hidden_dim': 128,
    'gnn_num_layers': 2,
    'gru_num_layers': 1,
    'gnn_dropout': 0.2,
    'gru_dropout': 0.2,
    'gnn_agg_method': 'mean',

    # Training
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'patience': 10,  # Early stopping patience

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def train_epoch(model, dataloader, optimizer, criterion, device, edge_index, graph_builder):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mae = 0
    total_valid = 0
    num_batches = 0

    for idx in range(len(dataloader.dataset)):
        # Get sample and target
        sample = dataloader.dataset[idx].to(device)
        target_timestep = idx + CONFIG['sequence_length']

        # Skip if target would be out of bounds
        if target_timestep >= len(graph_builder._travel_data):
            continue

        target = graph_builder._build_target_tensor(target_timestep).to(device)
        valid_mask = target != -1.0

        if valid_mask.sum() == 0:
            continue

        # Forward pass
        optimizer.zero_grad()
        output, _ = model(sample, edge_index)

        # Compute loss only on valid measurements
        loss = criterion(output[valid_mask], target[valid_mask].unsqueeze(1))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        with torch.no_grad():
            mae = torch.abs(output[valid_mask] - target[valid_mask].unsqueeze(1)).mean()

        total_loss += loss.item()
        total_mae += mae.item()
        total_valid += valid_mask.sum().item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    return avg_loss, avg_mae, total_valid // num_batches if num_batches > 0 else 0


def validate(model, dataloader, criterion, device, edge_index, graph_builder):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_mae = 0
    total_valid = 0
    num_batches = 0

    with torch.no_grad():
        for idx in range(len(dataloader.dataset)):
            sample = dataloader.dataset[idx].to(device)
            target_timestep = idx + CONFIG['sequence_length']

            if target_timestep >= len(graph_builder._travel_data):
                continue

            target = graph_builder._build_target_tensor(target_timestep).to(device)
            valid_mask = target != -1.0

            if valid_mask.sum() == 0:
                continue

            output, _ = model(sample, edge_index)
            loss = criterion(output[valid_mask], target[valid_mask].unsqueeze(1))
            mae = torch.abs(output[valid_mask] - target[valid_mask].unsqueeze(1)).mean()

            total_loss += loss.item()
            total_mae += mae.item()
            total_valid += valid_mask.sum().item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    return avg_loss, avg_mae, total_valid // num_batches if num_batches > 0 else 0


def main():
    print("=" * 70)
    print("TRAINING GRAPHSAGE-GRU MODEL FOR TRAVEL TIME PREDICTION")
    print("=" * 70)

    # ===== 1. Load Data =====
    print("\n[1/6] Loading data...")
    data_dir = Path("data")
    edge_files = [
        str(data_dir / "edge_data_day3.csv"),
        str(data_dir / "edge_data_day4.csv"),
        str(data_dir / "edge_data_day5.csv"),
        str(data_dir / "edge_data_day6.csv"),
        str(data_dir / "edge_data_day7.csv")
    ]

    edge_data = ",".join(edge_files)
    edge_connection = str(data_dir / "edge_connections.csv")
    osm = str(data_dir / "osm_roads_output.json")

    fileloader = FileLoader(edge_data, edge_connection, osm)
    graph_builder = GraphDatasetBuilder(fileloader)

    print(f"  ✓ Nodes (road segments): {graph_builder.x.shape[0]}")
    print(f"  ✓ Node features: {graph_builder.x.shape[1]}")
    print(f"  ✓ Graph edges: {graph_builder.edge_index.shape[1]}")
    print(f"  ✓ Timesteps: {len(graph_builder._travel_data)}")

    # ===== 2. Create Dataset =====
    print("\n[2/6] Creating temporal dataset...")
    temp_dataset = TemporalDatasetBuilder(graph_builder, CONFIG['sequence_length'])

    # Split into train/val
    train_size = int(CONFIG['train_split'] * len(temp_dataset))
    val_size = len(temp_dataset) - train_size

    train_dataset, val_dataset = random_split(
        temp_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  ✓ Total sequences: {len(temp_dataset)}")
    print(f"  ✓ Train sequences: {len(train_dataset)}")
    print(f"  ✓ Val sequences: {len(val_dataset)}")

    # Create "dataloaders" (we'll just use the datasets directly)
    train_loader = train_dataset
    val_loader = val_dataset

    # ===== 3. Initialize Model =====
    print("\n[3/6] Initializing model...")
    device = torch.device(CONFIG['device'])
    print(f"  Using device: {device}")

    sample = temp_dataset[0]
    in_dim = sample.shape[2]

    model = GraphSAGEGru(
        in_dim=in_dim,
        out_dim=1,
        gnn_num_layers=CONFIG['gnn_num_layers'],
        gru_num_layers=CONFIG['gru_num_layers'],
        gnn_hidden_dim=CONFIG['gnn_hidden_dim'],
        gru_hidden_dim=CONFIG['gru_hidden_dim'],
        gnn_dropout=CONFIG['gnn_dropout'],
        gru_dropout=CONFIG['gru_dropout'],
        gnn_agg_method=CONFIG['gnn_agg_method']
    ).to(device)

    edge_index = graph_builder.edge_index.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model parameters: {total_params:,}")

    # ===== 4. Setup Training =====
    print("\n[4/6] Setting up training...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    print(f"  ✓ Loss: MSE")
    print(f"  ✓ Optimizer: Adam (lr={CONFIG['learning_rate']})")
    print(f"  ✓ Scheduler: ReduceLROnPlateau")

    # ===== 5. Training Loop =====
    print("\n[5/6] Training...")
    print("-" * 70)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()

        # Train
        train_loss, train_mae, train_valid = train_epoch(
            model, train_loader, optimizer, criterion, device, edge_index, graph_builder
        )

        # Validate
        val_loss, val_mae, val_valid = validate(
            model, val_loader, criterion, device, edge_index, graph_builder
        )

        # Update scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} MAE: {train_mae:.2f}s | "
              f"Val Loss: {val_loss:.4f} MAE: {val_mae:.2f}s | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'config': CONFIG
            }, 'best_model.pth')
            print(f"  → Saved best model (val_loss: {val_loss:.4f}, mae: {val_mae:.2f}s)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # ===== 6. Final Evaluation =====
    print("\n[6/6] Final evaluation...")
    print("-" * 70)

    # Load best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on validation set
    val_loss, val_mae, val_valid = validate(
        model, val_loader, criterion, device, edge_index, graph_builder
    )

    print(f"\nBest Model Performance:")
    print(f"  Validation Loss (MSE): {val_loss:.4f}")
    print(f"  Validation MAE: {val_mae:.2f} seconds")
    print(f"  Avg valid measurements per sample: {val_valid}")
    print(f"  Best epoch: {checkpoint['epoch']+1}")

    # Sample predictions
    print("\nSample Predictions:")
    model.eval()
    with torch.no_grad():
        sample = temp_dataset[0].to(device)
        target = graph_builder._build_target_tensor(CONFIG['sequence_length']).to(device)
        output, _ = model(sample, edge_index)

        valid_mask = target != -1.0
        valid_indices = torch.where(valid_mask)[0][:5]

        for idx in valid_indices:
            pred = output[idx].item()
            actual = target[idx].item()
            error = abs(pred - actual)
            error_pct = 100 * error / actual if actual > 0 else 0
            print(f"  Node {idx.item()}: Pred={pred:.1f}s, Actual={actual:.1f}s, "
                  f"Error={error:.1f}s ({error_pct:.1f}%)")

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best model saved to: best_model.pth")


if __name__ == "__main__":
    main()
