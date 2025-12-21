import torch
from torch_geometric.data import Data
import pandas as pd
from typing import Dict
from src.data_manipulation.file_loader import FileLoader


class GraphDatasetBuilder:
    def __init__(self, loader: FileLoader, day_of_week_index: int,
                 x: torch.Tensor, edge_index: torch.Tensor,
                 edge_ids: torch.Tensor, edge_as_node_map: Dict[int, int]):

        # Daily Data
        self._travel_data = loader.get_travel_data()
        self._day_of_week_index = day_of_week_index

        # Static Graph Components (PASSED IN)
        self.x = x
        self.edge_index = edge_index
        self.edge_ids = edge_ids
        self._edge_as_node_map = edge_as_node_map

        # Dynamic/Daily Data Construction
        self.temporal_features = self._generate_temporal_features()

    def get_data(self):
        """
        Returns the static graph structure as a PyTorch Geometric Data object.
        Used primarily for accessing static components by the SequenceDataset.
        """
        return Data(x=self.x, edge_index=self.edge_index)

    def get_full_traffic_matrix(self):
        """
        Returns a tensor of shape (TimeSteps, Num_Nodes).
        Rows are timesteps, Columns are Nodes (sorted by _edge_as_node_map).
        """
        # 1. Identify travel time columns
        travel_columns = [c for c in self._travel_data.columns
                          if c.startswith("edge") and c.endswith("_traversal_time_sec")]

        # 2. Map column names to their assigned line graph node IDs
        col_to_node_id = {}
        for col in travel_columns:
            edge_id = int(col.split('_')[0].replace("edge", ""))
            if edge_id in self._edge_as_node_map:
                col_to_node_id[col] = self._edge_as_node_map[edge_id]

        # 3. Sort columns based on Node ID to ensure consistent order
        sorted_cols = sorted(col_to_node_id, key=col_to_node_id.get)

        # 4. Extract data and convert to tensor
        full_data = self._travel_data[sorted_cols].values
        return torch.tensor(full_data, dtype=torch.float)

    def _generate_temporal_features(self):
        """
        Generates time-of-day (sin/cos encoding) and day-of-week (one-hot) features
        for the entire travel data timeline. Output shape is (TimeSteps, 9).
        """
        df = self._travel_data.copy()
        num_timesteps = len(df)

        # Time-of-Day (Cyclic Encoding)
        # Total minutes in a day = 1440
        minutes_in_day_series = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute

        # Convert Pandas Series to Tensor for PyTorch operations
        minutes_in_day_tensor = torch.tensor(minutes_in_day_series.values, dtype=torch.float)

        # Sin/Cos encoding (2 dimensions)
        time_sin = torch.sin(2 * torch.pi * minutes_in_day_tensor / 1440).numpy()
        time_cos = torch.cos(2 * torch.pi * minutes_in_day_tensor / 1440).numpy()

        # Day-of-Week (One-Hot Encoding - 7 dimensions)
        # Create a series where every row has the same day index
        day_of_week_series = pd.Series(
            [self._day_of_week_index] * num_timesteps,
            index=df.index
        )

        # Convert to Categorical with all 7 categories explicitly defined
        day_of_week_categorical = pd.Categorical(
            day_of_week_series,
            categories=list(range(7))
        )

        # Apply one-hot encoding
        day_of_week_one_hot = pd.get_dummies(day_of_week_categorical, prefix='dow')

        # Concatenate Time features and Day features
        temporal_features_df = pd.DataFrame({
            'time_sin': time_sin,
            'time_cos': time_cos
        }, index=df.index)

        temporal_features_df = pd.concat([temporal_features_df, day_of_week_one_hot], axis=1).astype('float32')

        # Convert final DataFrame to Tensor (T, 9)
        return torch.tensor(temporal_features_df.values, dtype=torch.float)
