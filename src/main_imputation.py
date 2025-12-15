# src/main_imputation.py

import torch
import os, sys
import argparse
import pandas as pd
import numpy as np
# Adjusting imports since we are now in 'src/' instead of 'src/models/'
from .models.config import *
from .data_manipulation.file_loader import FileLoader
from .data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from .data_manipulation.data_pipeline import setup_data_loaders
from .models.GAT_BiGRU import GAT_BiGRU_Imputer
from .models.GraphSAGE_BiGRU import GraphSAGE_BiGRU_Imputer
from .data_manipulation.data_pipeline import setup_imputation_loader

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_MAP = {
    "GAT": GAT_BiGRU_Imputer,
    "GraphSAGE": GraphSAGE_BiGRU_Imputer
}

# The save directory paths are based on the project_root, which is correctly calculated above.
MODEL_LOAD_DIR = os.path.join(project_root, "src", "trained_models")
IMPUTATION_SAVE_DIR = os.path.join(project_root, "src", "results", "imputations")
os.makedirs(IMPUTATION_SAVE_DIR, exist_ok=True)


# ------------------

def load_trained_model(model_name, checkpoint_name, loader):
    """Instantiates a model and loads saved weights."""
    ModelClass = MODEL_MAP[model_name]
    sample_batch = next(iter(loader))
    static_feat_dim = sample_batch['x_static'].shape[2]

    # Determine GNN arguments based on model name
    if model_name == "GAT":
        gnn_hidden_arg = GAT_HIDDEN_DIM
    else:  # GraphSAGE
        gnn_out_dim = GAT_HIDDEN_DIM * GAT_HEADS
        gnn_hidden_arg = gnn_out_dim

    model = ModelClass(
        in_feat_static=static_feat_dim,
        in_feat_dynamic=1 + 9,
        gnn_hidden=gnn_hidden_arg,
        gru_hidden=GRU_HIDDEN_DIM,
        out_dim=1,
        heads=GAT_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    model_path = os.path.join(MODEL_LOAD_DIR, checkpoint_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # Set model to evaluation mode
    return model


def impute_and_save(model, all_data_loader, output_name):
    """
    Iterates over the entire dataset and generates the imputed values
    for all masked holes, saving the results to a CSV.
    """
    model.eval()
    all_results = []

    print("Starting imputation...")

    with torch.no_grad():
        for i, batch in enumerate(all_data_loader):
            # Move data to device
            x_dyn = batch['x_dynamic'].to(DEVICE)
            y_true = batch['y_true'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            x_stat = batch['x_static'].to(DEVICE)
            edge_index = batch['edge_index'].to(DEVICE)
            time_feats = batch['time_features'].to(DEVICE)

            # 1. Prediction (Imputation)
            prediction = model(x_dyn, x_stat, edge_index, time_feats)

            # 2. Extract only the imputed values (where mask is True)
            imputed_values = prediction[mask].cpu().numpy().flatten()
            true_values = y_true[mask].cpu().numpy().flatten()

            # 3. Get the indices/metadata for the imputed values
            B, N, T = mask.shape

            flat_mask = mask.view(-1)
            masked_flat_indices = torch.nonzero(flat_mask).flatten().cpu().numpy()

            # Convert flat index back to (batch_idx, node_idx, time_step)
            for idx, flat_idx in enumerate(masked_flat_indices):
                b = flat_idx // (N * T)
                n = (flat_idx % (N * T)) // T
                t = flat_idx % T

                # Retrieve the original edge ID from the batch metadata
                # Assuming edge_ids is part of the batch dict and has shape (B, N)
                edge_id = batch['edge_ids'][b, n].item()

                # Save the result
                all_results.append({
                    "edge_id": edge_id,
                    "time_step_idx": t,
                    "prediction": imputed_values[idx],
                    "true_value": true_values[idx]
                })

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(all_data_loader)} batches...")

    # 4. Convert to DataFrame and Save
    results_df = pd.DataFrame(all_results)

    # Save the imputed data to the results folder
    save_path = os.path.join(IMPUTATION_SAVE_DIR, f"{output_name}_imputed.csv")
    results_df.to_csv(save_path, index=False)

    print(f"\n--- Imputation Complete ---")
    print(f"Total imputed values saved: {len(results_df)}")
    print(f"Data saved to: {save_path}")


def main(model_name, run_name, checkpoint_name):
    # --- 1. DATA LOADING & PREPARATION ---
    DATA_ROOT = os.path.join(project_root, "src", TRAINING_DATA_FOLDER_NAME)

    # Data preparation uses ALL days for the imputation loader
    fileloader = FileLoader(
        edge_data_paths=(
                os.path.join(DATA_ROOT, "days", "edge_data_day3.csv") + "," +
                os.path.join(DATA_ROOT, "days", "edge_data_day4.csv") + "," +
                os.path.join(DATA_ROOT, "days", "edge_data_day5.csv") + "," +
                os.path.join(DATA_ROOT, "days", "edge_data_day6.csv") + "," +
                os.path.join(DATA_ROOT, "days", "edge_data_day7.csv")
        ),
        edge_connections_path=os.path.join(DATA_ROOT, "connections", "edge_connections.csv"),
        meta_data_path=os.path.join(DATA_ROOT, "osm_data", "osm_roads_output.json")
    )

    builder = GraphDatasetBuilder(loader=fileloader)

    # Use the full training set as the loader for imputation
    all_data_loader = setup_imputation_loader(
        builder, SEQ_LEN, BATCH_SIZE
    )

    # 2. --- LOAD MODEL ---
    model = load_trained_model(model_name, checkpoint_name, all_data_loader)

    # 3. --- RUN IMPUTATION AND SAVE ---
    impute_and_save(model, all_data_loader, run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run imputation on the full dataset using a trained GNN model.")

    parser.add_argument(
        '--model',
        type=str,
        default='GAT',
        choices=list(MODEL_MAP.keys()),
        help='The GNN model class used for training (GAT or GraphSAGE).'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='The filename of the saved model checkpoint (e.g., gs_h64_l002_exp1_best.pth).'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        required=True,
        help='A unique name for this imputation output file (e.g., gs_exp1_full_imputed).'
    )

    args = parser.parse_args()
    main(args.model, args.run_name, args.checkpoint)