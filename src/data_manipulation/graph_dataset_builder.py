import torch
from torch_geometric.data import Data
import pandas as pd
from collections import defaultdict
from src.data_manipulation.file_loader import FileLoader
from typing import Dict


class GraphDatasetBuilder:
    def __init__(self, loader: FileLoader, day_of_week_index: int,
                 x: torch.Tensor, edge_index: torch.Tensor,
                 edge_ids: torch.Tensor, edge_as_node_map: Dict[int, int]):  # ADDED STATIC ARGS

        # Daily Data
        self._travel_data = loader.get_travel_data()
        self._day_of_week_index = day_of_week_index

        # --- 1. Static Graph Components (PASSED IN) ---
        # These fields are NO LONGER calculated here.
        self.x = x
        self.edge_index = edge_index
        self.edge_ids = edge_ids
        self._edge_as_node_map = edge_as_node_map

        # --- 2. Dynamic/Daily Data Construction (REMAINS) ---
        # Build target vector (y) for the entire time span (not just timestep=0)
        # Note: We remove the 'timestep' argument from __init__ since we now work with the full matrix.
        # The first implementation of _build_target_tensor only returned one timestep,
        # so this is a placeholder adjustment to match the full matrix logic.

        # Build temporal features (time of day, day of week)
        self.temporal_features = self._generate_temporal_features()

        # NOTE: The self.y attribute (for a single timestep) is often not used
        # when full matrix methods are available. We rely on get_full_traffic_matrix().

    # --- REMOVED STATIC GRAPH BUILDING METHODS ---
    # The following methods are DELETED or MOVED to StaticGraphBuilder:
    # - _filter_adjacency_to_travel_edges
    # - _create_edge_as_node_map
    # - _build_line_graph_edge_index
    # - _build_node_features
    # ---------------------------------------------

    # --- UPDATED: _build_target_tensor (simplified, relies on static map) ---
    # NOTE: Since the full matrix is used by the dataloader, this method
    # might only be needed for debugging or single-timestep access.
    def _build_target_tensor(self, timestep=0):
        # Pick a single timestep from travel data
        row = self._travel_data.iloc[timestep]

        # Columns corresponding to edges
        travel_columns = [c for c in row.index
                          if c.startswith("edge") and c.endswith("_traversal_time_sec")]

        # Map travel times to edge_ids
        travel_dict = {}
        for c in travel_columns:
            edge_num_str = c.split('_')[0].replace("edge", "")
            edge_id = int(edge_num_str)
            travel_dict[edge_id] = row[c]

        # Build target tensor ordered by line graph node IDs
        y_list = []
        for edge_id in sorted(self._edge_as_node_map, key=lambda k: self._edge_as_node_map[k]):
            # Check for data presence. If column was present in master_cols but data is missing:
            if edge_id not in travel_dict:
                # The FileLoader should have ensured this column exists with a -1.0 sentinel
                # but we will stick to the previous implementation and trust the data is in the correct place
                # based on your previous error trace.

                # IMPORTANT: Due to the global column fix, this should rarely be a ValueError
                # if the FileLoader correctly sets missing values to -1.0.
                # For safety, use -1.0 if not found, as travel_dict should contain all columns
                # (even if they are -1.0 after FileLoader processing).
                y_list.append(travel_dict.get(edge_id, -1.0))
            else:
                y_list.append(travel_dict[edge_id])

        return torch.tensor(y_list, dtype=torch.float)

    # --- get_data (REVISED to use passed static data) ---
    def get_data(self):
        # Return as PyTorch Geometric Data object for the full time series
        # Note: Typically, Data objects hold single samples (N, F).
        # For time series, this often holds the N-dimensional static graph structure (x, edge_index).

        # Since the SequenceDataset handles the full time series, this is likely
        # only used for static representation:
        return Data(x=self.x, edge_index=self.edge_index)

        # --- get_full_traffic_matrix (REMAINS, relies on _edge_as_node_map) ---

    def get_full_traffic_matrix(self):
        """
        Returns a tensor of shape (TimeSteps, Num_Nodes)
        Rows are timesteps, Columns are Nodes (sorted by _edge_as_node_map)
        """
        # ... (Method logic remains the same, as it only uses self._travel_data and self._edge_as_node_map) ...
        # (See original code for full implementation - it is correct for the new structure)

        # Get all travel data columns
        travel_columns = [c for c in self._travel_data.columns
                          if c.startswith("edge") and c.endswith("_traversal_time_sec")]

        # Ensure columns are sorted by the node IDs you assigned
        col_to_node_id = {}
        for col in travel_columns:
            edge_id = int(col.split('_')[0].replace("edge", ""))
            # Use the global map created by StaticGraphBuilder
            if edge_id in self._edge_as_node_map:
                col_to_node_id[col] = self._edge_as_node_map[edge_id]
            # Else: If an edge is in the travel data but not in the map (shouldn't happen
            # if FileLoader uses master_cols), it is ignored.

        # 2. Sort columns based on Node ID
        sorted_cols = sorted(col_to_node_id, key=col_to_node_id.get)

        # 3. Extract data as tensor
        full_data = self._travel_data[sorted_cols].values
        return torch.tensor(full_data, dtype=torch.float)

    def _generate_temporal_features(self):
        """
        Generates time-of-day (sin/cos encoding) and day-of-week (one-hot) features
        for the entire travel data timeline, using the provided day_of_week_index.
        """
        df = self._travel_data.copy()
        num_timesteps = len(df)

        # Time-of-Day (Cyclic Encoding)
        # Total minutes in a day = 24 * 60 = 1440
        minutes_in_day_series = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute

        # *** FIX: Convert Pandas Series to Tensor ***
        minutes_in_day_tensor = torch.tensor(minutes_in_day_series.values, dtype=torch.float)

        # Sin/Cos encoding for periodicity (2 dimensions)
        # Now use the tensor for PyTorch operations:
        df['time_sin'] = torch.sin(2 * torch.pi * minutes_in_day_tensor / 1440).numpy()
        df['time_cos'] = torch.cos(2 * torch.pi * minutes_in_day_tensor / 1440).numpy()

        # --- Day-of-Week (File-Dependent One-Hot Encoding) ---
        # 1. Create a Series where every row has the same day index
        # This replaces the incorrect df['Timestamp'].dt.dayofweek call
        day_of_week_series = pd.Series(
            [self._day_of_week_index] * num_timesteps,
            index=df.index  # Ensure index alignment
        )

        # 2. Convert to Categorical with all 7 categories explicitly defined.
        day_of_week_categorical = pd.Categorical(
            day_of_week_series,
            categories=list(range(7))
        )

        # 3. Apply one-hot encoding
        day_of_week_one_hot = pd.get_dummies(day_of_week_categorical, prefix='dow')

        # Concatenate Time features (2D) and Day features (7D)
        temporal_features_df = pd.concat([df[['time_sin', 'time_cos']], day_of_week_one_hot], axis=1)

        temporal_features_df = temporal_features_df.astype('float32')

        # Convert final DataFrame to Tensor (T, 9)
        return torch.tensor(temporal_features_df.values, dtype=torch.float)
