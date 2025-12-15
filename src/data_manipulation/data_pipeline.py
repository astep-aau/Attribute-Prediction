# src/data_manipulation/data_pipeline.py (MODIFIED)

import random
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from src.data_manipulation.sequenceDataset import SequenceDataset

# ... (other imports)

# Update to accept a list of builders
def setup_data_loaders(builders_list, seq_len, mask_rate, batch_size):
    """
    Implements the Sparse, Seasonally-Balanced Split across multiple builders (days).
    The data from all builders is aggregated first, then the train/val/test split is applied.
    """

    # --- 1. DATA AGGREGATION ---
    # Concatenate the full traffic matrix (y_full) and temporal features (t_feats)
    # Static features (x, edge_index) are assumed constant across all days/builders

    # Check for empty list
    if not builders_list:
        raise ValueError("builders_list cannot be empty.")

    first_builder = builders_list[0]

    # 1.1 Aggregate Dynamic and Temporal Data
    traffic_matrices = [b.get_full_traffic_matrix() for b in builders_list]
    temporal_features = [b.temporal_features for b in builders_list]

    full_matrix = torch.cat(traffic_matrices, dim=0)
    full_temporal_features = torch.cat(temporal_features, dim=0)

    # 1.2 Validate Static Data Consistency (sanity check)
    # The first builder's static data will be used

    # This must be replaced with the full matrix length
    total_time_steps = len(full_matrix)

    # Calculate steps per day (T_day)
    # T_day is the length of the first traffic matrix
    T_day = len(traffic_matrices[0])
    num_days = len(builders_list)

    if total_time_steps != T_day * num_days:
        print(f"Warning: Data aggregation resulted in {total_time_steps} steps, expected {T_day * num_days}.")
        # Use T_day as the basis for indexing calculation

    # --- 2. TRAIN/VAL/TEST INDEXING (LOGIC REMAINS, BUT APPLIES TO CONCATENATED MATRIX) ---

    # Day indices: Day 3=0, Day 4=1, Day 5=2, Day 6=3, Day 7=4 (0-based file index)

    train_days_indices = [0, 2]  # Day 3, Day 5
    val_days_indices = [1, 3]  # Day 4, Day 6
    test_day_index = 4  # Day 7

    def _get_indices_for_days(day_indices, step_size):
        all_indices = []
        for idx in day_indices:
            start = idx * T_day
            end = start + T_day - seq_len + 1
            all_indices.extend(list(range(start, end, step_size)))
        return all_indices

    # Use sliding window (step=1) for train to maximize density
    train_indices = _get_indices_for_days(train_days_indices, step_size=1)
    random.shuffle(train_indices)

    # Use non-overlapping window (step=seq_len) for val/test
    val_indices = _get_indices_for_days(val_days_indices, step_size=seq_len)

    # Test indices (single day)
    test_indices = _get_indices_for_days([test_day_index], step_size=seq_len)

    # --- 3. CREATE DATASETS AND LOADERS (MODIFIED) ---

    print(f"\n--- Data Split Statistics ---")
    print(f"Total Time Steps: {total_time_steps}")
    print(f"Time Steps per Day (T): {T_day}")
    print(f"Train (Sat, Mon): {len(train_indices)} samples")
    print(f"Val   (Sun, Tue): {len(val_indices)} samples")
    print(f"Test  (Wed):      {len(test_indices)} samples")
    print("-----------------------------\n")

    # The SequenceDataset now needs to be initialized with the concatenated data,
    # and the static features/graph structure from the first builder.

    # NOTE: The SequenceDataset class must also be updated to accept the raw tensors
    # instead of a builder object, and only uses the builder for static elements.

    # A TEMPORARY ASSUMPTION IS MADE FOR SequenceDataset (requires internal modification):
    # The SequenceDataset must be updated to accept the full matrices/features directly.
    # We pass the full matrices and the static features from the first builder.

    def _create_dataset(indices):
        return SequenceDataset(
            # Pass all aggregated data
            data_matrix=full_matrix,
            temporal_features=full_temporal_features,
            # Pass static data from the first builder (assumed consistent)
            static_features=first_builder.x,
            edge_index=first_builder.edge_index,
            edge_ids=first_builder.edge_ids,

            seq_len=seq_len,
            mask_rate=mask_rate,
            active_indices=indices
        )

    train_set = _create_dataset(train_indices)
    val_set = _create_dataset(val_indices)
    test_set = _create_dataset(test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def setup_imputation_loader(builders_list, seq_len, batch_size):
    """
    Sets up a DataLoader for the ENTIRE dataset using the original
    missingness mask, without applying any additional random masking.
    """
    if not builders_list:
        raise ValueError("builders_list cannot be empty.")

    first_builder = builders_list[0]

    # Concatenate all builders' full datasets to create the single imputation dataset

    full_matrix = torch.cat([b.get_full_traffic_matrix() for b in builders_list], dim=0)
    full_temporal_features = torch.cat([b.temporal_features for b in builders_list], dim=0)
    total_time_steps = len(full_matrix)

    # Create sequential indices for the entire concatenated dataset
    full_indices = list(range(0, total_time_steps - seq_len + 1, seq_len))

    # The SequenceDataset needs the full matrices directly
    dataset = SequenceDataset(
        data_matrix=full_matrix,
        temporal_features=full_temporal_features,
        static_features=first_builder.x,
        edge_index=first_builder.edge_index,
        edge_ids=first_builder.edge_ids,
        seq_len=seq_len,
        mask_rate=0.0,  # No random masking for imputation
        active_indices=full_indices
    )

    # Create a DataLoader for the full dataset
    imputation_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be False to maintain sequential order for analysis
        drop_last=False
    )

    return imputation_loader