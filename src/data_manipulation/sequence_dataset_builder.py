import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, data_matrix: torch.Tensor, temporal_features: torch.Tensor,
                 static_features: torch.Tensor, edge_index: torch.Tensor,
                 edge_ids: torch.Tensor, seq_len: int = 12, mask_rate: float = 0.2,
                 active_indices=None):
        """
        Initializes the SequenceDataset. It uses pre-aggregated full matrices
        and static graph components to create sequential time windows.
        """
        self.seq_len = seq_len
        self.mask_rate = mask_rate

        self.traffic_matrix = data_matrix        # Full concatenated traffic data (T_total, N)
        self.temporal_features = temporal_features  # Full concatenated temporal features (T_total, F_temp)

        # Static/Graph features (assumed constant across days)
        self.static_features = static_features    # (N, F_stat)
        self.edge_index = edge_index              # (2, E)
        self.edge_ids = edge_ids                  # (N,)

        # Calculate All Block Start Indices
        total_time_steps = len(self.traffic_matrix)
        self.all_possible_starts = list(range(0, total_time_steps - self.seq_len + 1))

        # Determine Active Indices for Train/Val/Test split
        if active_indices is None:
            self.active_indices = self.all_possible_starts
        else:
            self.active_indices = active_indices

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, index):
        idx = self.active_indices[index]

        # 1. Slice the window (T, N)
        sequence_data = self.traffic_matrix[idx: idx + self.seq_len]
        sequence_temp_feats = self.temporal_features[idx: idx + self.seq_len]

        # 2. Determine Original Missing Holes (values of -1.0)
        # Sentinel value check
        original_hole_mask = (sequence_data < -0.99) & (sequence_data > -1.01)

        # 3. Initialize the Input Feature Matrix (x_dynamic)
        x_dynamic = sequence_data.clone()

        # 4. Initialize the Final Mask (This determines which points contribute to loss/metrics)
        final_mask = original_hole_mask.clone()

        # 5. Handle Masking
        if self.mask_rate > 0.0:
            # Training: Randomly mask GOOD data points, and only these points are in final_mask

            # Find all GOOD data points (non-holes)
            good_data_mask = ~original_hole_mask

            # Randomly select a percentage of the GOOD data points to mask
            random_mask_targets = torch.rand_like(sequence_data) < self.mask_rate
            random_training_mask = random_mask_targets & good_data_mask

            # The final mask (target) is ONLY the randomly generated mask
            final_mask = random_training_mask

            # The input mask combines original holes AND the newly random masked points
            input_mask = original_hole_mask | random_training_mask

            # Set all masked/missing points in the INPUT (x_dynamic) to 0.0
            x_dynamic[input_mask] = 0.0
        else:
            # Imputation/Testing: No random masking. final_mask remains original_hole_mask.
            # Set the original -1.0 holes in the INPUT (x_dynamic) to 0.0
            x_dynamic[original_hole_mask] = 0.0

        # 6. Create Target (y_true)
        y_true = sequence_data.clone()

        # --- CONSTRUCT X_FEAT (N, T, F_feat) for GNN Input ---

        # a. Dynamic Travel Time Input (F_dyn=1): (T, N) -> (N, T, 1)
        x_dynamic_t = x_dynamic.permute(1, 0).unsqueeze(-1)

        # b. Temporal Features (F_temp=9): (T, F_temp) -> (N, T, F_temp)
        # Repeat the time vector N times
        temp_feats_repeated = sequence_temp_feats.unsqueeze(0).repeat(self.static_features.shape[0], 1, 1)

        # c. Time-variant features: [Traffic, Temporal] (N, T, 10)
        x_time_variant = torch.cat([x_dynamic_t, temp_feats_repeated], dim=-1)

        # d. Static Features (F_stat=22): (N, F_stat) -> (N, 1, F_stat) -> (N, T, F_stat)
        # Expand static features across the T dimension
        x_static_expanded = self.static_features.unsqueeze(1).repeat(1, self.seq_len, 1)

        # e. Final Combined Features (N, T, F_feat=32)
        x_combined = torch.cat([x_time_variant, x_static_expanded], dim=-1)

        # f. Final Mask and Target (N, T) - Transpose to (N, T) from (T, N)
        final_mask = final_mask.permute(1, 0)
        final_y_true = y_true.permute(1, 0)

        return {
            'x_combined': x_combined,  # (N, T, 32)
            'edge_index': self.edge_index,  # (2, E)
            'y_true': final_y_true,  # (N, T)
            'mask': final_mask,  # (N, T)
        }