"""
Model configuration definitions for different architectures.
Add new model configs here to train different models.
"""

from src.models.graphsage_gru import GraphSAGEGru
from src.models.graphsage import GraphSAGE

# GraphSAGE-GRU Configuration
GRAPHSAGE_GRU_CONFIG = {
    'model_name': 'GraphSAGE-GRU',

    # Model factory - function that creates the model given in_dim
    'create_model': lambda in_dim: GraphSAGEGru(
        in_dim=in_dim,
        out_dim=1,
        gnn_hidden_dim=64,
        gru_hidden_dim=128,
        gnn_num_layers=2,
        gru_num_layers=1,
        gnn_dropout=0.2,
        gru_dropout=0.2,
        gnn_agg_method='mean'
    ),

    # Data
    'sequence_length': 5,
    'train_split': 0.8,

    # Training
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'patience': 10,
}

# GraphSAGE-GRU with higher capacity
GRAPHSAGE_GRU_LARGE_CONFIG = {
    'model_name': 'GraphSAGE-GRU-Large',

    # Model factory - function that creates the model given in_dim
    'create_model': lambda in_dim: GraphSAGEGru(
        in_dim=in_dim,
        out_dim=1,
        gnn_hidden_dim=128,
        gru_hidden_dim=256,
        gnn_num_layers=3,
        gru_num_layers=2,
        gnn_dropout=0.3,
        gru_dropout=0.3,
        gnn_agg_method='mean'
    ),

    # Data
    'sequence_length': 7,  # Longer sequences
    'train_split': 0.8,

    # Training
    'epochs': 50,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'patience': 15,
}

# Dictionary to easily access configs
MODEL_CONFIGS = {
    'graphsage_gru': GRAPHSAGE_GRU_CONFIG,
    'graphsage_gru_large': GRAPHSAGE_GRU_LARGE_CONFIG,
}
