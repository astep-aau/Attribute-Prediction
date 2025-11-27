"""
Generalized training script that works with multiple model types.
Usage: python train_generic.py --config graphsage_gru
       python train_generic.py --config graphsage_gru_large
       python train_generic.py --config graphsage_only
"""

from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.temporal_dataset_builder import TemporalDatasetBuilder
from src.training.train_utils import (
    load_model, train_epoch, validate,
    setup_optimizer_and_scheduler, evaluate_final_model
)
from configs.model_configs import MODEL_CONFIGS
from pathlib import Path
import torch
import torch.nn as nn
import time
import argparse
import json


def train_model(config, device='auto'):
    """
    Train a model with the given configuration.

    Args:
        config: Configuration dictionary from model_configs.py
        device: 'auto', 'cuda', or 'cpu'
    """
    print("=" * 80)
    print(f"TRAINING {config['model_name'].upper()} FOR TRAVEL TIME PREDICTION")
    print("=" * 80)

    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

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

    print(f"  Nodes (road segments): {graph_builder.x.shape[0]}")
    print(f"  Node features: {graph_builder.x.shape[1]}")
    print(f"  Graph edges: {graph_builder.edge_index.shape[1]}")
    print(f"  Timesteps: {len(graph_builder._travel_data)}")

    # ===== 2. Create Dataset =====
    print("\n[2/6] Creating temporal dataset...")
    temp_dataset = TemporalDatasetBuilder(graph_builder, config['sequence_length'])

    # Split into train/val using sequential split (temporal order preserved)
    train_size = int(config['train_split'] * len(temp_dataset))
    val_size = len(temp_dataset) - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(temp_dataset)))

    print(f"     Total sequences: {len(temp_dataset)}")
    print(f"     Train sequences: {len(train_indices)} (timesteps 0-{train_size-1})")
    print(f"     Val sequences: {len(val_indices)} (timesteps {val_size})")

    # ===== 3. Initialize Model =====
    print("\n[3/6] Initializing model...")
    print(f"  Using device: {device}")
    print(f"  Model: {config['model_name']}")

    sample = temp_dataset[0]
    in_dim = sample.shape[2]

    model = load_model(in_dim, config).to(device)
    edge_index = graph_builder.edge_index.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # ===== 4. Setup Training =====
    print("\n[4/6] Setting up training...")
    criterion = nn.MSELoss()
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model, config['learning_rate'], config['weight_decay']
    )

    print(f"    Loss: MSE")
    print(f"    Optimizer: Adam (lr={config['learning_rate']})")
    print(f"    Scheduler: ReduceLROnPlateau")
    print(f"\n  Anti-overfitting measures:")
    print(f"    - Train/Val split with NO temporal overlap")

    # Print dropout info if available
    dropout_info = []
    if 'gnn_dropout' in config['model_params']:
        dropout_info.append(f"{config['model_params']['gnn_dropout']} (GNN)")
    if 'gru_dropout' in config['model_params']:
        dropout_info.append(f"{config['model_params']['gru_dropout']} (GRU)")
    if 'dropout' in config['model_params']:
        dropout_info.append(f"{config['model_params']['dropout']}")

    if dropout_info:
        print(f"    - Dropout: {' & '.join(dropout_info)}")

    print(f"    - Weight decay (L2): {config['weight_decay']}")
    print(f"    - Early stopping patience: {config['patience']} epochs")
    print(f"    - Learning rate reduction on plateau")

    # ===== 5. Training Loop =====
    print("\n[5/6] Training...")
    print("-" * 80)

    best_val_loss = float('inf')
    patience_counter = 0
    train_history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Train
        train_loss, train_mae, train_valid = train_epoch(
            model, train_indices, temp_dataset, optimizer, criterion,
            device, edge_index, graph_builder, config['sequence_length']
        )

        # Validate
        val_loss, val_mae, val_valid = validate(
            model, val_indices, temp_dataset, criterion,
            device, edge_index, graph_builder, config['sequence_length']
        )

        # Update scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        # Track history
        train_history['train_loss'].append(train_loss)
        train_history['val_loss'].append(val_loss)
        train_history['train_mae'].append(train_mae)
        train_history['val_mae'].append(val_mae)

        # Calculate overfitting indicator
        loss_gap = val_loss - train_loss
        gap_pct = 100 * loss_gap / train_loss if train_loss > 0 else 0
        overfit_warning = " OVERFITTING" if gap_pct > 50 else ""

        # Print progress
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} MAE: {train_mae:.2f}s | "
              f"Val Loss: {val_loss:.4f} MAE: {val_mae:.2f}s | "
              f"Gap: {gap_pct:+.1f}%{overfit_warning} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Use config name from MODEL_CONFIGS as identifier
            config_name = [k for k, v in MODEL_CONFIGS.items() if v == config][0]
            model_filename = f"best_model_{config_name}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'config': config,
                'in_dim': in_dim
            }, model_filename)
            print(f"  → Saved best model to {model_filename} (val_loss: {val_loss:.4f}, mae: {val_mae:.2f}s)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # ===== 6. Final Evaluation =====
    print("\n[6/6] Final evaluation...")
    print("-" * 80)

    # Load best model
    config_name = [k for k, v in MODEL_CONFIGS.items() if v == config][0]
    model_filename = f"best_model_{config_name}.pth"
    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on validation set using shared utility
    val_loss, val_mae, val_valid = evaluate_final_model(
        model, val_indices, temp_dataset, criterion, device,
        edge_index, graph_builder, checkpoint, config
    )

    # Save training history
    config_name = [k for k, v in MODEL_CONFIGS.items() if v == config][0]
    history_filename = f"training_history_{config_name}.json"
    with open(history_filename, 'w') as f:
        json.dump(train_history, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best model saved to: {model_filename}")
    print(f"Training history saved to: {history_filename}")

    return model, train_history, checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train a model for travel time prediction')
    parser.add_argument('--config', type=str, default='graphsage_gru',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model configuration to use')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--list-configs', action='store_true',
                        help='List all available configurations')

    args = parser.parse_args()

    if args.list_configs:
        print("Available model configurations:")
        for key, config in MODEL_CONFIGS.items():
            print(f"  - {key}: {config['model_name']}")
        return

    config = MODEL_CONFIGS[args.config]
    train_model(config, device=args.device)


if __name__ == "__main__":
    main()
