# src/data_manipulation/data_pipeline.py

import random
import torch
from torch_geometric.loader import DataLoader
from src.data_manipulation.sequenceDataset import SequenceDataset


def setup_data_loaders(builders_list, seq_len, mask_rate, batch_size):
    """
    Implements the Sparse, Seasonally-Balanced Split across multiple builders (days).
    Data is aggregated first, then the train/val/test split indices are applied.
    """
    if not builders_list:
        raise ValueError("builders_list cannot be empty.")

    first_builder = builders_list[0]

    # 1. DATA AGGREGATION
    traffic_matrices = [b.get_full_traffic_matrix() for b in builders_list]
    temporal_features = [b.temporal_features for b in builders_list]

    full_matrix = torch.cat(traffic_matrices, dim=0)
    full_temporal_features = torch.cat(temporal_features, dim=0)

    total_time_steps = len(full_matrix)
    T_day = len(traffic_matrices[0])
    num_days = len(builders_list)

    if total_time_steps != T_day * num_days:
        print(f"Warning: Data aggregation resulted in {total_time_steps} steps, expected {T_day * num_days}.")

    # 2. TRAIN/VAL/TEST INDEXING
    train_days_indices = [0, 2]  # Day 3 (Sat), Day 5 (Mon)
    val_days_indices = [1, 3]    # Day 4 (Sun), Day 6 (Tue)
    test_day_index = 4           # Day 7 (Wed)

    def _get_indices_for_days(day_indices, step_size):
        all_indices = []
        for idx in day_indices:
            start = idx * T_day
            end = start + T_day - seq_len + 1
            all_indices.extend(list(range(start, end, step_size)))
        return all_indices

    # Use sliding window (step=1) for train
    train_indices = _get_indices_for_days(train_days_indices, step_size=1)
    random.shuffle(train_indices)

    # Use non-overlapping window (step=seq_len) for val/test
    val_indices = _get_indices_for_days(val_days_indices, step_size=seq_len)
    test_indices = _get_indices_for_days([test_day_index], step_size=seq_len)

    # 3. CREATE DATASETS AND LOADERS
    print(f"\n--- Data Split Statistics ---")
    print(f"Total Time Steps: {total_time_steps}")
    print(f"Time Steps per Day (T): {T_day}")
    print(f"Train (Sat, Mon): {len(train_indices)} samples")
    print(f"Val   (Sun, Tue): {len(val_indices)} samples")
    print(f"Test  (Wed):      {len(test_indices)} samples")
    print("-----------------------------\n")

    # Create a helper function to instantiate the SequenceDataset
    def _create_dataset(indices):
        return SequenceDataset(
            data_matrix=full_matrix,
            temporal_features=full_temporal_features,
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
    missingness mask (mask_rate=0.0) with non-overlapping windows.
    """
    if not builders_list:
        raise ValueError("builders_list cannot be empty.")

    first_builder = builders_list[0]

    # Concatenate all builders' full datasets
    full_matrix = torch.cat([b.get_full_traffic_matrix() for b in builders_list], dim=0)
    full_temporal_features = torch.cat([b.temporal_features for b in builders_list], dim=0)
    total_time_steps = len(full_matrix)

    # Create non-overlapping indices for the entire concatenated dataset
    full_indices = list(range(0, total_time_steps - seq_len + 1, seq_len))

    dataset = SequenceDataset(
        data_matrix=full_matrix,
        temporal_features=full_temporal_features,
        static_features=first_builder.x,
        edge_index=first_builder.edge_index,
        edge_ids=first_builder.edge_ids,
        seq_len=seq_len,
        mask_rate=0.0,
        active_indices=full_indices
    )

    imputation_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return imputation_loader