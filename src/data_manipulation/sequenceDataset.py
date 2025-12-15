import torch
from torch.utils.data import Dataset


# Removed: from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
# We no longer import the builder, but accept its output tensors directly.


class SequenceDataset(Dataset):
    def __init__(self, data_matrix: torch.Tensor, temporal_features: torch.Tensor,
                 static_features: torch.Tensor, edge_index: torch.Tensor,
                 edge_ids: torch.Tensor, seq_len: int = 12, mask_rate: float = 0.2,
                 active_indices=None):

        self.seq_len = seq_len
        self.mask_rate = mask_rate

        # --- MODIFIED: Get data directly as arguments ---
        self.traffic_matrix = data_matrix  # Full concatenated traffic data (T_total, N)
        self.temporal_features = temporal_features  # Full concatenated temporal features (T_total, F_temp)

        # Static/Graph features (assumed constant across days)
        self.static_features = static_features  # (N, F_stat)
        self.edge_index = edge_index  # (2, E)
        self.edge_ids = edge_ids  # (N,)
        # --- END MODIFIED ---

        # Calculate All Block Start Indices ---
        total_time_steps = len(self.traffic_matrix)

        # Calculate start indices for all possible overlapping blocks
        self.all_possible_starts = list(range(0, total_time_steps - self.seq_len + 1))

        # Calculate NON-OVERLAPPING start indices (for reference/testing only)
        # This is now less relevant as active_indices defines the sampling strategy
        self.all_block_starts = list(range(0, total_time_steps - self.seq_len + 1, self.seq_len))

        # Determine Active Indices ---
        if active_indices is None:
            # Default to all overlapping indices if none are passed
            self.active_indices = self.all_possible_starts
        else:
            # Use the specific indices passed from the main script (e.g., split indices)
            self.active_indices = active_indices

    def __len__(self):
        # The length is based on the currently active set of indices
        return len(self.active_indices)

    def __getitem__(self, index):
        # ... (Slicing and initial setup remains the same) ...

        idx = self.active_indices[index]

        # 1. Slice the window (T, N)
        sequence_data = self.traffic_matrix[idx: idx + self.seq_len]
        sequence_temp_feats = self.temporal_features[idx: idx + self.seq_len]

        # 2. Determine Original Missing Holes (values of -1.0)
        # Use a small tolerance for floating point safety
        original_hole_mask = (sequence_data < -0.99) & (sequence_data > -1.01)

        # 3. Initialize the Input Feature Matrix (x_dynamic)
        x_dynamic = sequence_data.clone()

        # 4. Initialize the Final Mask (This is the target for loss/evaluation)
        final_mask = original_hole_mask.clone()  # Default mask for imputation

        # 5. Handle Random Masking (Training Logic)
        if self.mask_rate > 0.0:
            # We want to randomly mask GOOD data points (non-holes)

            # Find all GOOD data points (where original_hole_mask is False)
            good_data_mask = ~original_hole_mask

            # Randomly select a percentage of the GOOD data points to mask
            random_mask_targets = torch.rand_like(sequence_data) < self.mask_rate

            # This is the mask containing ONLY the points we randomly selected
            random_training_mask = random_mask_targets & good_data_mask

            # CRITICAL TRAINING LOGIC:
            # During training (mask_rate > 0.0), the final mask is ONLY the random mask.
            final_mask = random_training_mask

            # Now, apply ALL masking (random + original holes) to the INPUT (x_dynamic)
            # Input mask includes original holes (which are now 0.0) AND the random training points
            input_mask = original_hole_mask | random_training_mask

            # Set all masked/missing points in the INPUT to 0.0 for the GNN
            x_dynamic[input_mask] = 0.0
        else:
            # Imputation Logic (mask_rate == 0.0)
            # The final_mask remains the original_hole_mask.
            # Set the original -1.0 holes in the INPUT to 0.0 for the GNN
            x_dynamic[original_hole_mask] = 0.0

        # 6. Create Target (y_true)
        # y_true is the ground truth travel time (same as sequence_data)
        y_true = sequence_data.clone()

        # --- NEW STEP 6: CONSTRUCT X_FEAT (N, T, F_feat) ---

        # a. Travel Time Input (Dynamic, F_dyn=1): (T, N) -> (N, T, 1)
        x_dynamic_t = x_dynamic.permute(1, 0).unsqueeze(-1)

        # b. Temporal Features (F_temp=9): (T, F_temp) -> (N, T, F_temp)
        # Repeat the sequence of temporal features (T) N times for the nodes
        temp_feats_repeated = sequence_temp_feats.unsqueeze(0).repeat(self.static_features.shape[0], 1, 1)

        # c. Time-variant features (N, T, 1 + F_temp)
        x_time_variant = torch.cat([x_dynamic_t, temp_feats_repeated], dim=-1)  # (N, T, 10)

        # d. Static Features (F_stat=22): (N, F_stat) -> (N, 1, F_stat)
        # Expand static features across the T dimension (T=12)
        x_static_expanded = self.static_features.unsqueeze(1).repeat(1, self.seq_len, 1)  # (N, 12, 22)

        # e. Final Combined Features (N, T, F_feat)
        # F_feat = 1 (Traffic) + 9 (Temporal) + 22 (Static) = 32
        x_combined = torch.cat([x_time_variant, x_static_expanded], dim=-1)  # (N, T, 32)
        # The order is now: [Traffic, Temporal_9D, Static_22D]

        # f. Final Mask and Target (N, T)
        final_mask = final_mask.permute(1, 0)
        final_y_true = y_true.permute(1, 0)

        return {
            'x_combined': x_combined,  # (N, T, 32)
            'edge_index': self.edge_index,  # (2, E)
            'y_true': final_y_true,  # (N, T)
            'mask': final_mask,  # (N, T)
        }