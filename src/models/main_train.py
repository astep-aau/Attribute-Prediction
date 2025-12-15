# main_train.py

import setproctitle

import logging
import os, sys
import argparse
from src.models.logging_utils import logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.config import *
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.data_manipulation.data_pipeline import setup_data_loaders
from src.models.GAT_BiGRU import GAT_BiGRU_Imputer
from src.models.trainer import run_training_and_testing
from src.models.GraphSAGE_BiGRU import GraphSAGE_BiGRU_Imputer
import pandas as pd
from src.data_manipulation.static_graph_builder import StaticGraphBuilder

MODEL_MAP = {
    "GAT": GAT_BiGRU_Imputer,
    "GraphSAGE": GraphSAGE_BiGRU_Imputer
}


def get_global_edge_columns_and_ids(file_paths):
    """
    Helper function to find the union of all edge IDs and their column names
    across all daily traffic files, ensuring a consistent node count (N).
    """
    all_edges_set = set()
    for path in file_paths:
        try:
            # We only read the first few rows for columns, as the data itself is large
            df = pd.read_csv(path, nrows=1)
            edge_cols = [c for c in df.columns if c.startswith("edge")]
            all_edges_set.update(edge_cols)
        except FileNotFoundError:
            logger.warning(f"Traffic file not found: {path}")

    # Remove the 'time_slot' column if it somehow got included
    all_edges_set.discard("time_slot")

    # 1. Get the list of physical IDs (int) used for StaticGraphBuilder
    # Example: ['edge1_...', 'edge5_...'] -> [1, 5, ...]
    global_edge_ids = sorted([int(c.split('_')[0].replace("edge", "")) for c in all_edges_set])

    # 2. Get the list of full column names (str) used for FileLoader consistency
    edges_sorted = sorted(all_edges_set, key=lambda x: int(x.split("_")[0].replace("edge", "")))
    master_cols = ["time_slot"] + edges_sorted

    return master_cols, global_edge_ids


def main(model_name, run_name, lr, gnn_dim, gru_dim, heads, dropout, gnn_layers):
    """
    Main entry point to run a single training/testing experiment.

    Args:
        model_name (str): The name of the GNN model to use ("GAT" or "GraphSAGE").
        run_name (str): The filename/identifier for the saved model and results.
        lr (float): Learning rate override.
        gnn_dim (int): GNN Hidden Dimension override.
        gru_dim (int): GRU Hidden Dimension override.
        heads (int): GAT Attention Heads override.
        dropout (float): Dropout rate override.
    """

    # Local variables for clarity
    use_model = model_name

    # --- USE OVERRIDE VALUES INSTEAD OF CONFIG CONSTANTS ---
    # These local variables are used to pass the parsed arguments to the model and trainer
    MODEL_LR = lr
    GNN_HIDDEN = gnn_dim
    GRU_HIDDEN = gru_dim
    GAT_ATTN_HEADS = heads
    MODEL_DROPOUT = dropout
    GNN_LAYERS = gnn_layers

    new_title = f"GNN_{model_name}_{run_name}"
    setproctitle.setproctitle(new_title)

    print(f"Using device: {DEVICE}")
    print(f"Saving model and results under run name: {run_name}")
    print(
        f"Hyperparameters: LR={MODEL_LR}, GNN_H={GNN_HIDDEN}, GRU_H={GRU_HIDDEN}, Heads={GAT_ATTN_HEADS}, Dropout={MODEL_DROPOUT}")

    # --- 1. DATA LOADING & PREPARATION (Unchanged) ---
    print("Loading raw data files...")

    DATA_ROOT = os.path.join(project_root, "src", TRAINING_DATA_FOLDER_NAME)

    FILE_CONFIG = [
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day3.csv"), "day_index": 5},  # Sat
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day4.csv"), "day_index": 6},  # Sun
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day5.csv"), "day_index": 0},  # Mon
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day6.csv"), "day_index": 1},  # Tue
        {"path": os.path.join(DATA_ROOT, "days", "edge_data_day7.csv"), "day_index": 2},  # Wed
    ]

    edge_connections_path = os.path.join(DATA_ROOT, "connections", "edge_connections.csv")
    meta_data_path = os.path.join(DATA_ROOT, "osm_data", "osm_roads_output.json")

    # --- PHASE 1: GLOBAL PRE-PROCESSING ---
    print("\n--- Phase 1: Determining Global Edge Set (for consistent node count) ---")
    file_paths = [cfg['path'] for cfg in FILE_CONFIG]
    global_master_cols, global_edge_ids = get_global_edge_columns_and_ids(file_paths)

    # --- PHASE 2: STATIC GRAPH BUILDING (ONE TIME) ---
    print("--- Phase 2: Building Static Graph Structure ---")

    # The loader needs a file path for edge_data_path, even if the StaticBuilder ignores its content
    static_loader = FileLoader(
        edge_data_path=FILE_CONFIG[0]['path'],  # Placeholder path
        edge_connections_path=edge_connections_path,
        meta_data_path=meta_data_path,
        # IMPORTANT: The static loader must still receive master_cols,
        # as the FileLoader constructor now requires it.
        master_cols=global_master_cols
    )

    static_builder = StaticGraphBuilder(static_loader, global_edge_ids)
    static_components = static_builder.get_static_components()

    # Log the resulting static graph size
    N_nodes = static_components['x'].shape[0]
    E_edges = static_components['edge_index'].shape[1]
    F_features = static_components['x'].shape[1]
    print(f"Static Graph Built: Nodes (N)={N_nodes}, Edges (E)={E_edges}, Static Features (F)={F_features}")

    # --- PHASE 3: DAILY DATA PROCESSING LOOP ---
    all_builders = []
    print("\n--- Phase 3: Processing Daily Data (Reusing Static Graph) ---")

    # Iterate through the file configurations to create a builder for each day
    for config in FILE_CONFIG:
        # 1. Create a FileLoader for the current day's travel data, ensuring consistent columns
        fileloader = FileLoader(
            edge_data_path=config['path'],
            edge_connections_path=edge_connections_path,
            meta_data_path=meta_data_path,
            master_cols=global_master_cols  # CRUCIAL: Fixes the N-mismatch error
        )

        # 2. Instantiate the GraphDatasetBuilder (now the Daily Data Processor)
        builder = GraphDatasetBuilder(
            loader=fileloader,
            day_of_week_index=config['day_index'],
            **static_components  # PASS ALL STATIC COMPONENTS
        )
        all_builders.append(builder)
        print(f"Processed file: {os.path.basename(config['path'])} (Day Index: {config['day_index']})")
    # --- 2. SETUP DATA LOADERS (Unchanged) ---
    train_loader, val_loader, test_loader = setup_data_loaders(
        all_builders,
        SEQ_LEN, MASK_RATE, BATCH_SIZE
    )

    # Calculate dynamic input dimensions using a sample batch
    sample_batch = next(iter(train_loader))
    static_feat_dim = sample_batch['x_static'].shape[2]

    TIME_FEAT_DIM = 9
    GAT_TOTAL_DYNAMIC_INPUT = 1 + TIME_FEAT_DIM

    print(f"Data ready. Train Batches: {len(train_loader)} | Nodes: {sample_batch['x_static'].shape[1]}")

    logger.info("Data loading and preparation complete. DataLoaders are ready.")
    logger.info(f"training on {DEVICE}")

    # --- 3. MODEL SETUP ---
    # GAT's true output dim is GNN_HIDDEN * HEADS
    GNN_OUT_DIM = GNN_HIDDEN * GAT_ATTN_HEADS

    # --- Dynamic Model Selection ---
    if use_model in MODEL_MAP:
        ModelClass = MODEL_MAP[use_model]
    else:
        raise ValueError(f"Invalid model selected: {use_model}. Must be one of {list(MODEL_MAP.keys())}")

    if use_model == "GAT":
        # GAT takes the HIDDEN size and multiplies by HEADS internally
        gnn_hidden_arg = GNN_HIDDEN
    elif use_model == "GraphSAGE":
        # GraphSAGE takes the full desired OUTPUT size
        gnn_hidden_arg = GNN_OUT_DIM

    print(f"--- STARTING EXPERIMENT using {use_model}-BiGRU ---")

    # --- PASS PARSED ARGUMENTS TO MODEL INITIALIZATION ---
    model = ModelClass(
        in_feat_static=static_feat_dim,
        in_feat_dynamic=GAT_TOTAL_DYNAMIC_INPUT,
        gnn_hidden=gnn_hidden_arg,
        gru_hidden=GRU_HIDDEN,
        out_dim=1,
        heads=GAT_ATTN_HEADS,
        dropout=MODEL_DROPOUT,
        num_gnn_layers=GNN_LAYERS
    ).to(DEVICE)

    # --- PASS PARSED ARGUMENT TO TRAINER ---
    # run_training_and_testing needs to be updated to accept LR
    run_training_and_testing(
        model,
        train_loader,
        val_loader,
        test_loader,
        use_model,
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

    # 1. Argument for the model choice (Existing)
    parser.add_argument(
        '--model',
        type=str,
        default='GAT',
        choices=list(MODEL_MAP.keys()),
        help='The GNN model to use (GAT or GraphSAGE). Defaults to GAT.'
    )

    # 2. Argument for the output file name (Existing)
    parser.add_argument(
        '--run_name',
        type=str,
        required=True,
        help='The unique name to use when saving the model checkpoints and results.'
    )

    # --- ADDED HYPERPARAMETER ARGUMENTS ---
    # These names match the arguments used in your bash script
    parser.add_argument(
        '--lr',
        type=float,
        default=LEARNING_RATE,  # Fallback to value from config.py
        help='The learning rate for the optimizer.'
    )
    parser.add_argument(
        '--gnn_dim',
        type=int,
        default=GAT_HIDDEN_DIM,  # Fallback to value from config.py
        help='The hidden dimension for the GNN layer (will be multiplied by heads for GAT).'
    )
    parser.add_argument(
        '--gru_dim',
        type=int,
        default=GRU_HIDDEN_DIM,  # Fallback to value from config.py
        help='The hidden dimension for the BiGRU layer.'
    )
    parser.add_argument(
        '--heads',
        type=int,
        default=GAT_HEADS,  # Fallback to value from config.py
        help='The number of attention heads (used by GAT).'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=DROPOUT,  # Fallback to value from config.py
        help='The dropout rate applied to the model.'
    )
    parser.add_argument(
        '--gnn_layers',
        type=int,
        default=1,  # Default is 1 layer if not specified
        help='The number of sequential GNN layers to stack (e.g., 2 or 3).'
    )
    # --- END ADDED ARGUMENTS ---

    # 3. Parse arguments and call main()
    args = parser.parse_args()

    # --- PASS ALL ARGUMENTS TO MAIN FUNCTION ---
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