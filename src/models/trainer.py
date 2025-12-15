# src/models/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
from src.models.config import *
from src.models.logging_utils import logger

# Define the project root path relative to this script's location (src/models)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Define the absolute paths for saving files
MODEL_SAVE_DIR = os.path.join(_project_root, "src", "trained_models")
LOG_SAVE_DIR = os.path.join(_project_root, "src", "logs_and_testing_results")

def train_epoch(model, loader, criterion, optimizer, device):
    """Runs a single training epoch."""

    model.train()
    total_loss = 0
    accumulate_steps = 8

    batch_start_time = time.time()
    batches_to_log = 10

    for i, batch in enumerate(loader):

        # Transpose x_dynamic from (B, N, T, F) to (B, T, N, F)
        # This matches the standard temporal model input (Batch, Time, Nodes, Features)
        x_dyn = batch['x_dynamic'].permute(0, 2, 1, 3).to(device)

        # Transpose y_true and mask from (B, N, T) to (B, T, N)
        # This ensures consistency for the loss calculation, assuming the model outputs (B, T, N).
        y_true = batch['y_true'].permute(0, 2, 1).to(device)
        mask = batch['mask'].permute(0, 2, 1).to(device)

        x_stat = batch['x_static'].to(device)

        edge_index = batch['edge_index'].to(device)

        time_feats = batch['time_features'].to(device)

        torch.cuda.synchronize()

        # 1. Forward Pass
        prediction = model(x_dyn, x_stat, edge_index, time_feats)

        # 2. Loss Calculation
        loss_matrix = criterion(prediction, y_true)

        # 3. Scale Loss: CRUCIAL STEP for accumulation
        masked_loss = loss_matrix[mask].mean() / accumulate_steps

        # 4. Backward Pass: Accumulate gradients
        masked_loss.backward()

        # 5. Optimization Step: Perform step only after 'accumulate_steps'
        if (i + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # Clear gradients after stepping

        if (i + 1) % batches_to_log == 0:
            current_time = time.time()
            time_elapsed_10_batches = current_time - batch_start_time
            avg_time_per_batch = time_elapsed_10_batches / batches_to_log

            # logger.info(
            #     f"Batch {i + 1}/{len(loader)} - "
            #     f"Avg Time per Batch over last {batches_to_log}: {avg_time_per_batch:.2f}s"
            # )

            # Reset the timer
            batch_start_time = current_time

        # Un-scale the loss for accurate logging
        total_loss += masked_loss.item() * accumulate_steps

    if len(loader) % accumulate_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader) if len(loader) > 0 else float('nan')


# ----------------------------------------------------
# Helper Function for MAPE (Mean Absolute Percentage Error)
# MAPE is calculated based on the masked, non-zero true values.
# ----------------------------------------------------
def calculate_mape_and_bias(prediction, y_true, mask):
    """Calculates MAPE and Bias for masked predictions."""

    # 1. Mask Tensors
    y_true_masked = y_true[mask]
    prediction_masked = prediction[mask]

    # 2. MAPE Calculation (Requires true values to be non-zero)
    # We add a small epsilon to prevent division by zero in case of noise.
    epsilon = 1e-5

    # | (y_true - pred) / y_true |
    percentage_error = torch.abs((y_true_masked - prediction_masked) / (y_true_masked + epsilon))
    mape = (100 * percentage_error.mean()).item()  # Convert to percentage and scalar

    # 3. Bias Calculation (Average over- or under-prediction)
    # (Prediction - True Value) / Number of Samples
    bias = (prediction_masked - y_true_masked).mean().item()

    return mape, bias

def evaluate(model, loader, criterion, device):
    """Evaluates the model on validation or test data."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            # Move data to device and preprocess (same logic as before)
            x_dyn = batch['x_dynamic'].permute(0, 2, 1, 3).to(device)
            y_true = batch['y_true'].permute(0, 2, 1).to(device)
            mask = batch['mask'].permute(0, 2, 1).to(device)
            x_stat = batch['x_static'].to(device)
            edge_index = batch['edge_index'].to(device)
            time_feats = batch['time_features'].to(device)

            prediction = model(x_dyn, x_stat, edge_index, time_feats)
            loss_matrix = criterion(prediction, y_true)

            masked_loss = loss_matrix[mask].mean()

            if torch.isnan(masked_loss):
                continue

            total_loss += masked_loss.item()

    return total_loss / len(loader) if len(loader) > 0 else float('nan')


def run_training_and_testing(model, train_loader, val_loader, test_loader,
                             gnn_model_name, run_name, learning_rate,
                             gnn_hidden_dim, gru_hidden_dim, gat_heads,
                             dropout_rate, gnn_layers):
    """Coordinates the full training, validation, and testing cycle."""

    start_time_total = time.time()  # timer for total training time

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='none')
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    val_loss_history = []  # To track overfitting gap

    # --- TRAINING AND VALIDATION LOOP ---
    for epoch in range(NUM_EPOCHS):
        start_time_epoch = time.time()

        # Train
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        avg_val_loss = evaluate(model, val_loader, criterion, DEVICE)
        val_loss_history.append(avg_val_loss)  # Track validation loss

        epoch_time = time.time() - start_time_epoch

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] ({epoch_time:.2f}s) | "
            f"Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}")

        logger.info(
            f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] ({epoch_time:.2f}s) | "
            f"Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}"
        )



        if patience_counter >= PATIENCE:
            print(f"Stopping early! Validation loss hasn't improved for {PATIENCE} epochs.")
            break

    total_training_time = time.time() - start_time_total  # <-- Stop timer
    print("\nTraining complete.")

    # --- FINAL TESTING PHASE ---
    print("\nStarting final testing on the dedicated day...")

    if best_model_state:
        model.load_state_dict(best_model_state)

        # 1. Run Evaluation
        avg_test_loss_mse = evaluate(model, test_loader, criterion, DEVICE)

        # 2. Calculate RMSE
        test_rmse = avg_test_loss_mse ** 0.5

        # 3. Calculate MAPE and Bias
        # Run one final loop to accumulate data for MAPE and Bias
        total_mape = 0
        total_bias = 0
        num_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                # Transpose x_dynamic from (B, N, T, F) to (B, T, N, F)
                x_dyn = batch['x_dynamic'].permute(0, 2, 1, 3).to(DEVICE)

                # Transpose y_true and mask from (B, N, T) to (B, T, N)
                y_true = batch['y_true'].permute(0, 2, 1).to(DEVICE)
                mask = batch['mask'].permute(0, 2, 1).to(DEVICE)

                x_stat = batch['x_static'].to(DEVICE)
                edge_index = batch['edge_index'].to(DEVICE)
                time_feats = batch['time_features'].to(DEVICE)

                prediction = model(x_dyn, x_stat, edge_index, time_feats)

                batch_mape, batch_bias = calculate_mape_and_bias(prediction, y_true, mask)

                total_mape += batch_mape
                total_bias += batch_bias
                num_batches += 1

        final_mape = total_mape / num_batches
        final_bias = total_bias / num_batches

        print(f"   Testing Complete on Independent Day. Final Test Loss (MSE): {avg_test_loss_mse:.6f}")
        print(f"   Root Mean Squared Error (RMSE): {test_rmse:.6f}")
        print(f"   Mean Absolute Percentage Error (MAPE): {final_mape:.2f}%")
        print(f"   Bias (Avg Over/Under-Prediction): {final_bias:.6f}")

        # --- SAVE RESULTS AND MODEL ---

        overfitting_gap = avg_train_loss - best_val_loss

        # 1. Prepare JSON Data
        results_data = {
            "model_name": run_name,
            "gnn_model_used": gnn_model_name,
            "hyperparameters": {
                    "SEQ_LEN": SEQ_LEN,
                    "BATCH_SIZE": BATCH_SIZE,
                    "MASK_RATE": MASK_RATE,
                    "LEARNING_RATE": learning_rate,
                    "GNN_HIDDEN_DIM": gnn_hidden_dim,
                    "GRU_HIDDEN_DIM": gru_hidden_dim,
                    "GNN_LAYERS": gnn_layers,
                    "GRU_LAYERS": GRU_LAYERS,
                    "GAT_HEADS": gat_heads,
                    "DROPOUT": dropout_rate,
                    "PATIENCE": PATIENCE
            },
            "metrics": {
                "test_mse": avg_test_loss_mse,
                "test_rmse": test_rmse,
                "test_mape": final_mape,
                "bias": final_bias,
                "overfitting_gap_val_diff": overfitting_gap
            },
            "Best_epoch_metrics": {
                "Best_epoch_train_loss_mse": avg_train_loss,
                "Best_epoch_val_loss_mse": avg_val_loss,
            },
            "timing": {
                "total_training_time_s": total_training_time,
                "epochs_completed": epoch + 1
            }
        }

        # 2. Save JSON Result
        log_filename = f"{run_name}_results.json"
        log_path = os.path.join(LOG_SAVE_DIR, log_filename)

        with open(log_path, 'w') as f:
            json.dump(results_data, f, indent=4)

        print(f"Results saved to {log_path}")

        # 3. Save Model Checkpoint
        model_filename = f"{run_name}.pth"
        save_path = os.path.join(MODEL_SAVE_DIR, model_filename)

        torch.save(best_model_state, save_path)
        print(f"Model saved successfully to {save_path}")
    else:
        print("Warning: Training failed or was interrupted. No model state to save.")