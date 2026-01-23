"""
Chengdu Imputation & Evaluation
Runs imputation on the converted Didi Chengdu dataset with random missingness (10/20/30%)
and reports MAE, RMSE, and MAPE for each checkpoint.
"""
import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

# --- PATH SETUP ---
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.config import *
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.data_manipulation.static_graph_builder import StaticGraphBuilder
from src.data_manipulation.sequence_dataset_builder import SequenceDataset
from src.models.GAT_BiGRU import GAT_BiGRU_Imputer
from src.models.GraphSAGE_BiGRU import GraphSAGE_BiGRU_Imputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = project_root / 'src' / 'TrainingData' / 'didi_chengdu_converted'
CHECKPOINT_DIR = project_root / 'src' / 'app' / 'saved_models'
RESULTS_PATH = project_root / 'imputation_evaluation_results_chengdu.csv'


def get_global_edge_columns_and_ids(file_paths: List[Path]) -> Tuple[List[str], List[int]]:
    """Get global edge set from all files."""
    all_edges_set = set()
    for path in file_paths:
        try:
            df = pd.read_csv(path, nrows=1)
            edge_cols = [c for c in df.columns if c.startswith("edge") and c.endswith("sec")]
            all_edges_set.update(edge_cols)
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")

    global_edge_ids = sorted([int(c.split('_')[0].replace("edge", "")) for c in all_edges_set])
    edges_sorted = sorted(all_edges_set, key=lambda x: int(x.split("_")[0].replace("edge", "")))
    master_cols = ["time_slot"] + edges_sorted

    return master_cols, global_edge_ids


def parse_hyperparams_from_filename(filename: str):
    """Parse hyperparameters from model checkpoint filename."""
    try:
        layers = int(re.search(r'_L(\d+)', filename).group(1))
        gnn_dim = int(re.search(r'_GNN(\d+)', filename).group(1))
        gru_dim = int(re.search(r'_GRU(\d+)', filename).group(1))
        heads = int(re.search(r'_H(\d+)', filename).group(1))
        dropout_match = re.search(r'_D([\d.]+)', filename)
        dropout = float(dropout_match.group(1)) if dropout_match else 0.3
        return layers, gnn_dim, gru_dim, heads, dropout
    except Exception:
        logger.warning(f"Could not parse hyperparams from {filename}, using defaults")
        return 1, 200, 200, 1, 0.2


def apply_random_mask(data_tensor: torch.Tensor, mask_rate: float):
    """Apply random mask to data tensor and return masked data plus mask."""
    valid_mask = (data_tensor != -1.0)
    num_valid = valid_mask.sum().item()
    num_to_mask = int(num_valid * mask_rate)

    valid_indices = torch.nonzero(valid_mask, as_tuple=True)
    perm = torch.randperm(num_valid)[:num_to_mask]
    mask_indices = tuple(idx[perm] for idx in valid_indices)

    mask = torch.zeros_like(data_tensor, dtype=torch.bool)
    mask[mask_indices] = True

    masked_data = data_tensor.clone()
    masked_data[mask] = -1.0

    return masked_data, mask


def evaluate_imputation(model, test_loader, mask_rate: float, device):
    """Evaluate model with given mask rate; returns MAE, RMSE, MAPE, count."""
    model.eval()

    all_true_values = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x_combined = batch['x_combined'].to(device)  # (B, T, N, F)
            edge_index = batch['edge_index'].to(device)

            y_original = x_combined[:, :, :, 0]  # (B, T, N)

            y_masked, mask = apply_random_mask(y_original, mask_rate)

            x_masked = x_combined.clone()
            x_masked[:, :, :, 0] = y_masked

            x_input = x_masked.permute(0, 2, 1, 3)
            predictions = model(x_input, edge_index)  # (B, N, T)

            mask_permuted = mask.permute(0, 2, 1)
            y_original_permuted = y_original.permute(0, 2, 1)

            if mask_permuted.sum() > 0:
                true_vals = y_original_permuted[mask_permuted].cpu().numpy()
                pred_vals = predictions[mask_permuted].cpu().numpy()

                all_true_values.extend(true_vals)
                all_predictions.extend(pred_vals)

            if batch_idx % 50 == 0:
                logger.info(f"  Processed batch {batch_idx}/{len(test_loader)}")

    all_true_values = np.array(all_true_values)
    all_predictions = np.array(all_predictions)

    mae = np.mean(np.abs(all_true_values - all_predictions))
    rmse = np.sqrt(np.mean((all_true_values - all_predictions) ** 2))

    non_zero_mask = all_true_values != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((all_true_values[non_zero_mask] - all_predictions[non_zero_mask]) / all_true_values[non_zero_mask])) * 100
    else:
        mape = np.nan

    return mae, rmse, mape, len(all_true_values)


def run_chengdu_imputation():
    logger.info(f"Using device: {DEVICE}")

    if not DATA_ROOT.exists():
        logger.error(f"Converted data not found at {DATA_ROOT}")
        logger.error("Please run convert_didi_to_harbin.py first!")
        return

    daily_files = sorted((DATA_ROOT / 'days').glob('edge_data_day*.csv'))
    if not daily_files:
        logger.error("No daily data files found!")
        return

    test_file = daily_files[-1]
    logger.info(f"Using {test_file.name} for evaluation")

    edge_conn_path = DATA_ROOT / 'connections' / 'edge_connections.csv'
    meta_path = DATA_ROOT / 'osm_data' / 'osm_roads_output.json'

    master_cols, global_edge_ids = get_global_edge_columns_and_ids([test_file])
    logger.info(f"Total edges: {len(global_edge_ids)}")

    static_loader = FileLoader(str(test_file), str(edge_conn_path), str(meta_path), master_cols)
    static_builder = StaticGraphBuilder(static_loader, global_edge_ids)
    static_components = static_builder.get_static_components()

    builder = GraphDatasetBuilder(static_loader, day_of_week_index=0, **static_components)
    full_matrix = builder.get_full_traffic_matrix()
    full_temporal_features = builder.temporal_features

    total_time_steps = len(full_matrix)
    num_samples = max(0, total_time_steps - SEQ_LEN + 1)
    active_indices = list(range(0, num_samples, SEQ_LEN))

    dataset = SequenceDataset(
        data_matrix=full_matrix,
        temporal_features=full_temporal_features,
        static_features=static_components['x'],
        edge_index=static_components['edge_index'],
        edge_ids=static_components['edge_ids'],
        seq_len=SEQ_LEN,
        mask_rate=0.0,
        active_indices=active_indices
    )

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.info(f"Dataset ready: {len(dataset)} samples, batch_size={BATCH_SIZE}")

    sample = next(iter(test_loader))
    gnn_input_dim = sample['x_combined'].shape[3]
    logger.info(f"Model input features: {gnn_input_dim}")

    checkpoints = list(CHECKPOINT_DIR.glob('*.pth'))
    if not checkpoints:
        logger.error(f"No checkpoints found in {CHECKPOINT_DIR}")
        return

    mask_rates = [0.1, 0.2, 0.3]
    results = []

    for ckpt_path in checkpoints:
        model_name = ckpt_path.stem
        logger.info("\n" + "=" * 80)
        logger.info(f"Evaluating: {model_name}")
        logger.info("=" * 80)

        is_gat = 'GAT' in model_name
        ModelClass = GAT_BiGRU_Imputer if is_gat else GraphSAGE_BiGRU_Imputer

        layers, gnn_dim, gru_dim, heads, dropout = parse_hyperparams_from_filename(model_name)
        logger.info(f"Config: layers={layers}, gnn_dim={gnn_dim}, gru_dim={gru_dim}, heads={heads}, dropout={dropout}")

        if is_gat:
            model = ModelClass(
                in_feat=gnn_input_dim,
                gnn_hidden=gnn_dim,
                gru_hidden=gru_dim,
                out_dim=1,
                heads=heads,
                dropout=dropout,
                num_gnn_layers=layers
            ).to(DEVICE)
        else:
            model = ModelClass(
                in_feat=gnn_input_dim,
                gnn_hidden=gnn_dim * heads,
                gru_hidden=gru_dim,
                out_dim=1,
                heads=heads,
                dropout=dropout,
                num_gnn_layers=layers
            ).to(DEVICE)

        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            continue

        for mask_rate in mask_rates:
            logger.info(f"\nEvaluating with {int(mask_rate*100)}% missingness...")
            try:
                mae, rmse, mape, num_masked = evaluate_imputation(model, test_loader, mask_rate, DEVICE)
                logger.info(f"Results: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, Masked values={num_masked}")

                results.append({
                    'model': model_name,
                    'mask_rate': f"{int(mask_rate*100)}%",
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'num_masked': num_masked
                })
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80 + "\n")
    logger.info(results_df.to_string(index=False))
    logger.info(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == '__main__':
    run_chengdu_imputation()
