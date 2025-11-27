"""
Shared utility functions for training models.
Contains common functions used by both train_generic.py and train_batch.py.
"""

import torch
import torch.nn as nn


def load_model(in_dim, config):
    """
    Load model from config.

    Args:
        in_dim: Input feature dimension
        config: Configuration dictionary with 'create_model' factory function

    Returns:
        Instantiated model
    """
    return config['create_model'](in_dim)


def train_epoch(model, indices, temp_dataset, optimizer, criterion, device, edge_index, graph_builder, sequence_length):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mae = 0
    total_valid = 0
    num_batches = 0

    for idx in indices:
        # Get sample and target
        sample = temp_dataset[idx].to(device)
        target_timestep = idx + sequence_length

        # Skip if target would be out of bounds
        if target_timestep >= len(graph_builder._travel_data):
            continue

        target = graph_builder._build_target_tensor(target_timestep).to(device)
        valid_mask = target != -1.0

        if valid_mask.sum() == 0:
            continue

        # Forward pass
        optimizer.zero_grad()

        # Handle both temporal models (return tuple) and non-temporal (return single tensor)
        output = model(sample, edge_index)
        if isinstance(output, tuple):
            output = output[0]  # Get predictions from (predictions, hidden_state)

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


def validate(model, indices, temp_dataset, criterion, device, edge_index, graph_builder, sequence_length):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_mae = 0
    total_valid = 0
    num_batches = 0

    with torch.no_grad():
        for idx in indices:
            sample = temp_dataset[idx].to(device)
            target_timestep = idx + sequence_length

            if target_timestep >= len(graph_builder._travel_data):
                continue

            target = graph_builder._build_target_tensor(target_timestep).to(device)
            valid_mask = target != -1.0

            if valid_mask.sum() == 0:
                continue

            # Handle both temporal models (return tuple) and non-temporal (return single tensor)
            output = model(sample, edge_index)
            if isinstance(output, tuple):
                output = output[0]

            loss = criterion(output[valid_mask], target[valid_mask].unsqueeze(1))
            mae = torch.abs(output[valid_mask] - target[valid_mask].unsqueeze(1)).mean()

            total_loss += loss.item()
            total_mae += mae.item()
            total_valid += valid_mask.sum().item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    return avg_loss, avg_mae, total_valid // num_batches if num_batches > 0 else 0


def setup_optimizer_and_scheduler(model, learning_rate, weight_decay):
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: The model to optimize
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight

    Returns:
        optimizer, scheduler tuple
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    return optimizer, scheduler


def evaluate_final_model(model, val_indices, temp_dataset, criterion, device,
                         edge_index, graph_builder, checkpoint, config):
    """
    Perform final evaluation and print analysis.

    Args:
        model: Trained model
        val_indices: Validation indices
        temp_dataset: Temporal dataset
        criterion: Loss criterion
        device: Device to run on
        edge_index: Graph edge index
        graph_builder: Graph dataset builder
        checkpoint: Saved checkpoint dict
        config: Model configuration

    Returns:
        val_loss, val_mae, val_valid tuple
    """
    # Evaluate on validation set
    val_loss, val_mae, val_valid = validate(
        model, val_indices, temp_dataset, criterion,
        device, edge_index, graph_builder, config['sequence_length']
    )

    print(f"\nBest Model Performance:")
    print(f"  Model: {config['model_name']}")
    print(f"  Validation Loss (MSE): {val_loss:.4f}")
    print(f"  Validation MAE: {val_mae:.2f} seconds")
    print(f"  Training Loss (MSE): {checkpoint['train_loss']:.4f}")
    print(f"  Train/Val Gap: {100*(val_loss-checkpoint['train_loss'])/checkpoint['train_loss']:+.1f}%")
    print(f"  Avg valid measurements per sample: {val_valid}")
    print(f"  Best epoch: {checkpoint['epoch']+1}")

    # Overfitting analysis
    final_gap = 100 * (val_loss - checkpoint['train_loss']) / checkpoint['train_loss']
    print(f"\n  Overfitting Analysis:")
    if final_gap < 10:
        print(f"    ✓ Excellent: Train/Val gap is only {final_gap:.1f}%")
    elif final_gap < 30:
        print(f"    ✓ Good: Some generalization gap ({final_gap:.1f}%) but acceptable")
    elif final_gap < 50:
        print(f"         Moderate overfitting detected ({final_gap:.1f}% gap)")
        print(f"       Consider: increase dropout, add more regularization")
    else:
        print(f"        Significant overfitting ({final_gap:.1f}% gap)")
        print(f"       Consider: increase dropout, reduce model size, more data")

    # Sample predictions
    print("\nSample Predictions:")
    model.eval()
    with torch.no_grad():
        sample = temp_dataset[0].to(device)
        target = graph_builder._build_target_tensor(config['sequence_length']).to(device)

        output = model(sample, edge_index)
        if isinstance(output, tuple):
            output = output[0]

        valid_mask = target != -1.0
        valid_indices = torch.where(valid_mask)[0][:5]

        for idx in valid_indices:
            pred = output[idx].item()
            actual = target[idx].item()
            error = abs(pred - actual)
            error_pct = 100 * error / actual if actual > 0 else 0
            print(f"  Node {idx.item()}: Pred={pred:.1f}s, Actual={actual:.1f}s, "
                  f"Error={error:.1f}s ({error_pct:.1f}%)")

    return val_loss, val_mae, val_valid
