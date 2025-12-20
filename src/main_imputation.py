import torch
import os
import glob
import pandas as pd
import numpy as np
import sys
import logging
import re

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.config import *
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.data_manipulation.data_pipeline import setup_data_loaders
from src.data_manipulation.static_graph_builder import StaticGraphBuilder
from src.models.GAT_BiGRU import GAT_BiGRU_Imputer
from src.models.GraphSAGE_BiGRU import GraphSAGE_BiGRU_Imputer

# Configuration
CHECKPOINT_DIR = os.path.join(project_root, "src", "trained_models", "pth")
DATA_ROOT = os.path.join(project_root, "src", "TrainingData")
OUTPUT_DIR = os.path.join(project_root, "imputed_days")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info(f"Project Root: {project_root}")
logger.info(f"Data Root: {DATA_ROOT}")
logger.info(f"Checkpoint Dir: {CHECKPOINT_DIR}")


def get_global_info():
    """Determines the global edge set based on Day 7 to avoid missing file errors."""
    test_path = os.path.join(DATA_ROOT, "days", "edge_data_day7.csv")

    if not os.path.exists(test_path):
        logger.error(f"Test file not found at: {test_path}")
        raise FileNotFoundError(test_path)

    logger.info(f"Reading global info from: {test_path}")
    df = pd.read_csv(test_path, nrows=1)
    edge_cols = [c for c in df.columns if c.startswith("edge") and c.endswith("sec")]

    global_edge_ids = sorted([int(c.split('_')[0].replace("edge", "")) for c in edge_cols])
    edges_sorted = sorted(edge_cols, key=lambda x: int(x.split("_")[0].replace("edge", "")))
    master_cols = ["time_slot"] + edges_sorted

    logger.info(f"Detected {len(global_edge_ids)} edges/nodes.")
    return master_cols, global_edge_ids


def parse_hyperparams_from_name(run_name):
    """
    Parses the filename string to extract architectural hyperparameters.
    Format expected: ..._L{layers}_LR{lr}_GNN{gnn}_GRU{gru}_H{heads}_D{dropout}
    """
    try:
        # Regex to find patterns like L3, GNN300, GRU300, H1
        layers = int(re.search(r'_L(\d+)', run_name).group(1))
        gnn_dim = int(re.search(r'_GNN(\d+)', run_name).group(1))
        gru_dim = int(re.search(r'_GRU(\d+)', run_name).group(1))
        heads = int(re.search(r'_H(\d+)', run_name).group(1))

        # Dropout often has a dot, e.g., D0.2
        dropout_match = re.search(r'_D([\d.]+)', run_name)
        dropout = float(dropout_match.group(1)) if dropout_match else 0.3

        return layers, gnn_dim, gru_dim, heads, dropout
    except (AttributeError, ValueError) as e:
        logger.warning(f"Could not parse params from {run_name}, using config defaults. Error: {e}")
        return 1, GAT_HIDDEN_DIM, GRU_HIDDEN_DIM, GAT_HEADS, DROPOUT


def run_standalone_inference():
    master_cols, global_edge_ids = get_global_info()

    test_file_path = os.path.join(DATA_ROOT, "days", "edge_data_day7.csv")
    edge_conn_path = os.path.join(DATA_ROOT, "connections", "edge_connections.csv")
    meta_path = os.path.join(DATA_ROOT, "osm_data", "osm_roads_output.json")

    logger.info("Initializing Data Pipeline...")
    static_loader = FileLoader(test_file_path, edge_conn_path, meta_path, master_cols)
    static_builder = StaticGraphBuilder(static_loader, global_edge_ids)
    static_components = static_builder.get_static_components()

    builder = GraphDatasetBuilder(static_loader, day_of_week_index=2, **static_components)

    from src.data_manipulation.data_pipeline import setup_imputation_loader
    logger.info("Setting up Imputation DataLoader...")
    test_loader = setup_imputation_loader([builder], SEQ_LEN, batch_size=1)

    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    if not checkpoint_files:
        logger.warning(f"No .pth files found in {CHECKPOINT_DIR}")
        return

    logger.info(f"Found {len(checkpoint_files)} checkpoints. Starting Batch Inference...")

    for pth_path in checkpoint_files:
        run_name = os.path.splitext(os.path.basename(pth_path))[0]
        is_gat = "GAT" in run_name

        # --- DYNAMIC PARSING STEP ---
        layers, gnn_dim, gru_dim, heads, dropout = parse_hyperparams_from_name(run_name)
        logger.info(f"--- Processing Model: {run_name} ---")
        logger.info(f"Parsed Config: Layers={layers}, GNN_Dim={gnn_dim}, GRU_Dim={gru_dim}, Heads={heads}")

        sample_batch = next(iter(test_loader))
        gnn_in = sample_batch['x_combined'].shape[3]

        ModelClass = GAT_BiGRU_Imputer if is_gat else GraphSAGE_BiGRU_Imputer

        # Reconstruct the exact architecture used during training
        model = ModelClass(
            in_feat=gnn_in,
            gnn_hidden=gnn_dim,  # Use parsed dim
            gru_hidden=gru_dim,  # Use parsed dim
            out_dim=1,
            heads=heads,  # Use parsed heads
            dropout=dropout,  # Use parsed dropout
            num_gnn_layers=layers  # Use parsed layer count
        ).to(DEVICE)

        # This should now succeed without "size mismatch" errors
        model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
        model.eval()
        all_results = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x_combined = batch['x_combined'].to(DEVICE)
                X_feat_input = x_combined.permute(0, 2, 1, 3)
                y_true = batch['y_true'].permute(0, 2, 1).to(DEVICE)
                mask = batch['mask'].permute(0, 2, 1).to(DEVICE)
                edge_index = batch['edge_index'].to(DEVICE)

                prediction = model(X_feat_input, edge_index)

                final_vals = y_true.clone()
                final_vals[mask] = prediction[mask]
                all_results.append(final_vals.cpu().numpy())

                if (i + 1) % 50 == 0:
                    logger.info(f"   Batch {i + 1}/{len(test_loader)} processed...")

                # --- Updated Export Logic ---
                # full_data shape: (Total_Batches, Batch_Size, T, N)
                # After concatenate: (Total_Time_Steps, N)
                full_data = np.concatenate(all_results, axis=0)
                B_total, T, N = full_data.shape
                reshaped_values = full_data.reshape(B_total * T, N)

                # 1. Retrieve the original time_slots from the FileLoader
                # We need to ensure we align the imputed data with the correct timestamps
                original_df = static_loader.get_travel_data()

                # Note: Depending on your SEQ_LEN, you may need to slice the original_df
                # to match the windowing used in the DataLoader
                # For SEQ_LEN=1, the indices usually match directly.
                time_slots = original_df["time_slot"].values[:reshaped_values.shape[0]]

                # 2. Create the DataFrame with the correct headers
                # master_cols[1:] contains the edge column names (excluding "time_slot")
                out_df = pd.DataFrame(reshaped_values, columns=master_cols[1:])

                # 3. Insert the time_slot column back at the start
                out_df.insert(0, "time_slot", time_slots)

                # 4. Save with headers and no index to match input structure
                output_path = os.path.join(OUTPUT_DIR, f"{run_name}.csv")
                out_df.to_csv(output_path, index=False)

                logger.info(f"SUCCESS: Exported {output_path} with shape {out_df.shape}")

if __name__ == "__main__":
    try:
        run_standalone_inference()
    except Exception as e:
        logger.exception("Inference failed with an error:")