from src.data_manipulation.data_loader import DataLoader
import torch
from torch_geometric.data import Data
import pandas as pd
from collections import defaultdict

class GraphDatasetBuilder:
    def __init__(self, loader: DataLoader, day_of_week_index: int, timestep: int = 0):
        self._adjacency_df = loader.get_adjacency()
        self._travel_data = loader.get_travel_data()
        self._meta_data_df = loader.get_meta_data()
        self._day_of_week_index = day_of_week_index

        # Filter adjacency to only include edges that have travel data
        self._filter_adjacency_to_travel_edges()

        # Map each edge to a consecutive line-graph node ID
        self._edge_as_node_map = self._create_edge_as_node_map()

        # Build line graph connectivity
        self.edge_index = self._build_line_graph_edge_index()

        self.edge_ids = torch.tensor(
            sorted(self._edge_as_node_map, key=lambda k: self._edge_as_node_map[k]),
            dtype=torch.long
        )

        # Build node features (per edge)
        self.x = self._build_node_features()

        # Build target vector (travel time) for the specified timestep
        self.y = self._build_target_tensor(timestep=timestep)

        # Build temporal features (time of day, day of week)
        self.temporal_features = self._generate_temporal_features()

    def _filter_adjacency_to_travel_edges(self):
        """
        Filter adjacency to only include edges that have travel data.
        In a line graph, each road segment (edge) becomes a node.
        """
        # Get edge_ids from travel data columns
        travel_columns = [c for c in self._travel_data.columns
                         if c.startswith("edge") and c.endswith("_traversal_time_sec")]

        travel_edge_ids = set()
        for col in travel_columns:
            # Parse "edge123_traversal_time_sec" -> 123
            edge_num_str = col.split('_')[0].replace("edge", "")
            travel_edge_ids.add(int(edge_num_str))

        original_count = len(self._adjacency_df)
        self._adjacency_df = self._adjacency_df[self._adjacency_df["edge_id"].isin(travel_edge_ids)].copy()

        print(f"Filtered from {original_count} edges to {len(self._adjacency_df)} edges with travel data")

    def _create_edge_as_node_map(self):
        # Map original edges to consecutive node IDs
        return {old_edge_id: new_id for new_id, old_edge_id in enumerate(self._adjacency_df["edge_id"])}

    def _build_line_graph_edge_index(self):
        """
        Build line graph edge connectivity.

        directed edge A -> edge B exists if A's end vertex = B's start vertex
        (traffic can flow from road A into road B)
        """
        # Directed: A -> B only if A ends where B starts
        vertex_end_to_edges = defaultdict(list)  # Edges that START at this vertex
        vertex_start_from_edges = {}  # Where each edge ENDS

        for _, row in self._adjacency_df.iterrows():
            edge_id = row["edge_id"]
            vertex_end_to_edges[row["vertex_start_id"]].append(edge_id)
            vertex_start_from_edges[edge_id] = row["vertex_end_id"]

        line_edges = []
        for edge_from, vertex_end in vertex_start_from_edges.items():
            # Find all edges that start where edge_from ends
            for edge_to in vertex_end_to_edges.get(vertex_end, []):
                if edge_from != edge_to:  # No self-loops
                    n1 = self._edge_as_node_map[edge_from]
                    n2 = self._edge_as_node_map[edge_to]
                    line_edges.append((n1, n2))

        if not line_edges:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(list(zip(*line_edges)), dtype=torch.long)


    def _build_node_features(self):
        """
        Build the node features:
            road_type : string -> one hot encoded
            oneway : Boolean -> int
        """
        sorted_df = self._adjacency_df.sort_values(
            by='edge_id',
            key=lambda col: col.map(self._edge_as_node_map)
        )

        road_types = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                      "residential", "motorway_link", "trunk_link", "primary_link", "secondary_link",
                      "tertiary_link", "living_street", "service", "pedestrian", "track", "bus_guideway",
                      "escape", "raceway", "road", "busway"]

        # Build lookup by iterating ways first
        edge_to_way = {}
        for way in self._meta_data_df.itertuples():
            nodes_set = set(way.nodes)

            # Check which edges belong to this way
            for row in sorted_df.itertuples():
                start_str = str(row.vertex_start_id)
                end_str = str(row.vertex_end_id)

                if start_str in nodes_set and end_str in nodes_set:
                    edge_to_way[row.edge_id] = way

        feature_list = []
        missing_count = 0
        unknown_road_types = set()
        for row in sorted_df.itertuples():
            if row.edge_id in edge_to_way:
                way = edge_to_way[row.edge_id]
                oneway = 1 if way.oneway else 0
                type_encoded = [1 if way.road_type == rt else 0 for rt in road_types]

                # Track unknown road types
                if way.road_type not in road_types:
                    unknown_road_types.add(way.road_type)

                feature_vector = type_encoded + [oneway]
                feature_list.append(feature_vector)
            else:
                # Default features if no way found
                feature_list.append([0] * (len(road_types) + 1))
                #print(f"No meta data for edge: {row.edge_id}")
                missing_count += 1

        print(f"{missing_count} edges are missing metadata out of {len(sorted_df)}")
        if unknown_road_types:
            print(f"Warning: Found unknown road types: {unknown_road_types}")
        return torch.tensor(feature_list, dtype=torch.float)

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
            if edge_id not in travel_dict:
                raise ValueError(f"Edge {edge_id} missing travel data at timestep {timestep}")
            y_list.append(travel_dict[edge_id])

        return torch.tensor(y_list, dtype=torch.float)

    def get_data(self):
        # Return as PyTorch Geometric Data object
        return Data(x=self.x, edge_index=self.edge_index, y=self.y)

    def get_full_traffic_matrix(self):
        """
        Returns a tensor of shape (TimeSteps, Num_Nodes)
        Rows are timesteps, Columns are Nodes (sorted by _edge_as_node_map)
        """
        # Get all travel data columns
        travel_columns = [c for c in self._travel_data.columns
                          if c.startswith("edge") and c.endswith("_traversal_time_sec")]

        # Ensure columns are sorted by the node IDs you assigned
        # 1. Create a map of {column_name: node_id}
        col_to_node_id = {}
        for col in travel_columns:
            edge_id = int(col.split('_')[0].replace("edge", ""))
            if edge_id in self._edge_as_node_map:
                col_to_node_id[col] = self._edge_as_node_map[edge_id]

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

    def get_full_dataset(self, seq_len: int, mask_rate: float, apply_split: bool):
        """
        Creates the full dataset of graph sequences for training/evaluation.

        This method is now used for both training (with split/masking)
        and imputation (without split/masking).
        """

        from .sequenceDataset import SequenceDataset
        # NOTE: Since you don't have the implementation for the SequenceDataset
        # class (which takes seq_len, mask_rate), I will assume its logic.

        # 1. Get the full traffic matrix (Num_Timesteps x Num_Nodes)
        full_traffic_matrix = self.get_full_traffic_matrix()

        # 2. Get the static features (Num_Nodes x Num_Features)
        static_features = self.x

        # 3. Get temporal features (Num_Timesteps x Num_Temporal_Features)
        temporal_features = self._generate_temporal_features()

        # 4. Get connectivity
        edge_index = self.edge_index

        # 5. Get edge IDs (for saving imputation results)
        edge_ids = torch.tensor(
            sorted(self._edge_as_node_map, key=lambda k: self._edge_as_node_map[k]),
            dtype=torch.long
        )

        full_dataset = SequenceDataset(
            data_matrix=full_traffic_matrix,
            static_features=static_features,
            temporal_features=temporal_features,
            edge_index=edge_index,
            edge_ids=edge_ids,
            seq_len=seq_len,
            mask_rate=mask_rate,
            split_ratio=(0, 0, 1) if not apply_split else (0.7, 0.15, 0.15)  # Default split ratios
        )

        if apply_split:
            # Assumes SequenceDataset returns split loaders if apply_split=True
            return full_dataset.train_set, full_dataset.val_set, full_dataset.test_set
        else:
            # If not split, return the entire dataset
            return full_dataset