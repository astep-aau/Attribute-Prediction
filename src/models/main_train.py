# main_train.py

import setproctitle
import os, sys
import argparse
import pandas as pd

# ----------------- PATH SETUP -----------------
# Ensure project root is in system path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# ----------------- CUSTOM IMPORTS -----------------
from src.models.logging_utils import logger
from src.models.config import *
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.data_manipulation.data_pipeline import setup_data_loaders
from src.data_manipulation.static_graph_builder import StaticGraphBuilder
from src.models.GAT_BiGRU import GAT_BiGRU_Imputer
from src.models.GraphSAGE_BiGRU import GraphSAGE_BiGRU_Imputer
from src.models.trainer import run_training_and_testing

MODEL_MAP = {
    "GAT": GAT_BiGRU_Imputer,
    "GraphSAGE": GraphSAGE_BiGRU_Imputer
}


def get_global_edge_columns_and_ids(file_paths):
    """
    Finds the union of all edge IDs and their column names across all daily traffic files,
    ensuring a consistent node count (N) across the entire dataset.

    Returns:
        Tuple[List[str], List[int]]: (master_cols, global_edge_ids)
    """
    all_edges_set = set()
    for path in file_paths:
        try:
            df = pd.read_csv(path, nrows=1)
            edge_cols = [c for c in df.columns if c.startswith("edge") and c.endswith("sec")]
            all_edges_set.update(edge_cols)
        except FileNotFoundError:
            logger.warning(f"Traffic file not found: {path}")

    # 1. Get the list of physical IDs (int) for StaticGraphBuilder
    global_edge_ids = sorted([int(c.split('_')[0].replace("edge", "")) for c in all_edges_set])

    # 2. Get the list of full column names (str) for FileLoader consistency
    edges_sorted = sorted(all_edges_set, key=lambda x: int(x.split("_")[0].replace("edge", "")))
    master_cols = ["time_slot"] + edges_sorted

    return master_cols, global_edge_ids


def main(model_name, run_name, lr, gnn_dim, gru_dim, heads, dropout, gnn_layers):
    """
    Main entry point to run a single training/testing experiment.
    """
    MODEL_LR = lr
    GNN_HIDDEN = gnn_dim
    GRU_HIDDEN = gru_dim
    GAT_ATTN_HEADS = heads
    MODEL_DROPOUT = dropout
    GNN_LAYERS = gnn_layers

    setproctitle.setproctitle(f"GNN_{model_name}_{run_name}")

    print(f"Using device: {DEVICE}")
    print(f"Saving model and results under run name: {run_name}")
    print(
        f"Hyperparameters: LR={MODEL_LR}, GNN_H={GNN_HIDDEN}, GRU_H={GRU_HIDDEN}, Heads={GAT_ATTN_HEADS}, Dropout={MODEL_DROPOUT}, GNN_Layers={GNN_LAYERS}")

    # --- 1. CONFIGURATION ---
    DATA_ROOT = os.path.join(project_root, "src", TRAINING_DATA_FOLDER_NAME)

    FILE_CONFIG = [
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day3.csv"), "day_index": 5},  # Sat (Train)
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day4.csv"), "day_index": 6},  # Sun (Val)
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day5.csv"), "day_index": 0},  # Mon (Train)
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day6.csv"), "day_index": 1},  # Tue (Val)
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day7.csv"), "day_index": 2},  # Wed (Test)
    ]
    edge_connections_path = os.path.join(DATA_ROOT, "connections", "edge_connections.csv")
    meta_data_path = os.path.join(DATA_ROOT, "osm_data", "osm_roads_output.json")

    # --- 2. GLOBAL PRE-PROCESSING AND STATIC GRAPH BUILDING ---
    print("\n--- Phase 1: Determining Global Edge Set ---")
    file_paths = [cfg['path'] for cfg in FILE_CONFIG]
    global_master_cols, global_edge_ids = get_global_edge_columns_and_ids(file_paths)

    print("--- Phase 2: Building Static Graph Structure ---")
    static_loader = FileLoader(
        edge_data_path=FILE_CONFIG[0]['path'],
        edge_connections_path=edge_connections_path,
        meta_data_path=meta_data_path,
        master_cols=global_master_cols
    )
    static_builder = StaticGraphBuilder(static_loader, global_edge_ids)
    static_components = static_builder.get_static_components()

    N_nodes = static_components['x'].shape[0]
    E_edges = static_components['edge_index'].shape[1]
    F_features = static_components['x'].shape[1]
    print(f"Static Graph Built: Nodes (N)={N_nodes}, Edges (E)={E_edges}, Static Features (F)={F_features}")

    # --- 3. DAILY DATA PROCESSING LOOP ---
    all_builders = []
    print("\n--- Phase 3: Processing Daily Data ---")

    for config in FILE_CONFIG:
        fileloader = FileLoader(
            edge_data_path=config['path'],
            edge_connections_path=edge_connections_path,
            meta_data_path=meta_data_path,
            master_cols=global_master_cols
        )

        builder = GraphDatasetBuilder(
            loader=fileloader,
            day_of_week_index=config['day_index'],
            **static_components
        )
        all_builders.append(builder)
        print(f"Processed file: {os.path.basename(config['path'])} (Day Index: {config['day_index']})")

    # --- 4. SETUP DATA LOADERS ---
    train_loader, val_loader, test_loader = setup_data_loaders(
        all_builders,
        SEQ_LEN, MASK_RATE, BATCH_SIZE
    )

    # Determine input dimensions from a sample batch
    sample_batch = next(iter(train_loader))
    X_COMBINED = sample_batch['x_combined']
    total_feat_dim = X_COMBINED.shape[3]
    GNN_INPUT_DIM = total_feat_dim

    print(f"Data ready. Train Batches: {len(train_loader)} | Nodes: {X_COMBINED.shape[1]}")
    logger.info("Data loading and preparation complete. DataLoaders are ready.")
    logger.info(f"training on {DEVICE}")

    # --- 5. MODEL SETUP ---
    ModelClass = MODEL_MAP.get(model_name)
    if not ModelClass:
        raise ValueError(f"Invalid model selected: {model_name}. Must be one of {list(MODEL_MAP.keys())}")

    print(f"--- STARTING EXPERIMENT using {model_name}-BiGRU ---")

    # Configure GNN hidden argument based on model type
    gnn_hidden_arg = GNN_HIDDEN * GAT_ATTN_HEADS if model_name == "GraphSAGE" else GNN_HIDDEN

    model = ModelClass(
        in_feat=GNN_INPUT_DIM,
        gnn_hidden=gnn_hidden_arg,
        gru_hidden=GRU_HIDDEN,
        out_dim=1,
        heads=GAT_ATTN_HEADS,
        dropout=MODEL_DROPOUT,
        num_gnn_layers=GNN_LAYERS
    ).to(DEVICE)

    # --- 6. RUN TRAINING AND TESTING ---
    run_training_and_testing(
        model,
        train_loader,
        val_loader,
        test_loader,
        model_name,
        run_name,
        MODEL_LR,
        GNN_HIDDEN,
        GRU_HIDDEN,
        GAT_ATTN_HEADS,
        MODEL_DROPOUT,
        GNN_LAYERS
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spatio-Temporal Graph Imputation Model.")

    parser.add_argument(
        '--model',
        type=str,
        default='GAT',
        choices=list(MODEL_MAP.keys()),
        help='The GNN model to use (GAT or GraphSAGE). Defaults to GAT.'
    )

    parser.add_argument(
        '--run_name',
        type=str,
        required=True,
        help='The unique name to use when saving the model checkpoints and results.'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=LEARNING_RATE,
        help='The learning rate for the optimizer.'
    )
    parser.add_argument(
        '--gnn_dim',
        type=int,
        default=GAT_HIDDEN_DIM,
        help='The base hidden dimension for the GNN layer.'
    )
    parser.add_argument(
        '--gru_dim',
        type=int,
        default=GRU_HIDDEN_DIM,
        help='The hidden dimension for the BiGRU layer.'
    )
    parser.add_argument(
        '--heads',
        type=int,
        default=GAT_HEADS,
        help='The number of attention heads (used by GAT).'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=DROPOUT,
        help='The dropout rate applied to the model.'
    )
    parser.add_argument(
        '--gnn_layers',
        type=int,
        default=1,
        help='The number of sequential GNN layers to stack.'
    )

    args = parser.parse_args()

    main(
        args.model,
        args.run_name,
        args.lr,
        args.gnn_dim,
        args.gru_dim,
        args.heads,
        args.dropout,
        args.gnn_layers
    )