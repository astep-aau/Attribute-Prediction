# src/data_manipulation/static_graph_builder.py

import torch
import pandas as pd
from typing import Dict, Tuple, List
from src.data_manipulation.file_loader import FileLoader  # Used to load metadata/adjacency
from collections import defaultdict

class StaticGraphBuilder:
    def __init__(self, loader: FileLoader, global_edge_ids: List[int]):
        """
        Initializes the builder with static road network data and the global set of edges.

        Args:
            loader: A FileLoader instance providing access to adjacency and metadata.
            global_edge_ids: A sorted list of all unique edge IDs found across all traffic data files.
        """
        self._adj_data = loader.get_adjacency()  # Edge connections
        self._meta_data = loader.get_meta_data()  # OSM road features
        self._global_edge_ids = global_edge_ids

        # 1. Map physical edge IDs to sequential node IDs (0 to N-1)
        self.edge_as_node_map, self.edge_ids = self._create_edge_as_node_map()

        # 2. Build Static Node Features (x)
        self.x = self._build_node_features()

        # 3. Build Adjacency Matrix (edge_index)
        self.edge_index = self._build_line_graph_edge_index()

    def _create_edge_as_node_map(self) -> Tuple[Dict[int, int], torch.Tensor]:
        """
        Creates a mapping from the physical road ID (int) to the graph's sequential node ID (0 to N-1).
        N is the total number of unique edges in the dataset.
        """

        # Use the global set of edge IDs provided to define the nodes (N)
        sorted_edge_ids = sorted(self._global_edge_ids)

        edge_as_node_map = {}
        for i, edge_id in enumerate(sorted_edge_ids):
            edge_as_node_map[edge_id] = i

        # The node IDs (used for indexing) are simply 0 to N-1
        node_ids_tensor = torch.tensor(sorted_edge_ids, dtype=torch.int)

        return edge_as_node_map, node_ids_tensor

    def _build_node_features(self) -> torch.Tensor:
        """
        Generates the static feature tensor 'x' (N, F_static) for all nodes
        using the complex topological matching from the old GraphDatasetBuilder.
        """

        road_types = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                      "residential", "motorway_link", "trunk_link", "primary_link", "secondary_link",
                      "tertiary_link", "living_street", "service", "pedestrian", "track", "bus_guideway",
                      "escape", "raceway", "road", "busway"]

        # 1. Filter adjacency data to only include the global set of nodes (self._global_edge_ids)
        # This acts like the old _filter_adjacency_to_travel_edges, but uses the global set.
        filtered_adj_df = self._adj_data[
            self._adj_data["edge_id"].isin(self.edge_as_node_map.keys())
        ].copy()

        # Sort the filtered adjacency DF by the new node ID for correct feature ordering
        sorted_df = filtered_adj_df.sort_values(
            by='edge_id',
            key=lambda col: col.map(self.edge_as_node_map)
        )

        # 2. Build topological lookup (edge_to_way mapping from the old code)
        edge_to_way = {}

        # Ensure metadata is in a format that supports the 'itertuples' attributes (nodes, oneway, road_type)
        # The index of self._meta_data is the way ID, not the edge ID!
        for way in self._meta_data.itertuples():
            # Check if way.nodes is iterable (it should be a list/set of strings for vertex IDs)
            if hasattr(way, 'nodes'):
                nodes_set = set(way.nodes)
            else:
                # Handle case where 'nodes' column might be missing or empty
                continue

            # Check which edges belong to this way
            for row in sorted_df.itertuples():
                # Vertex IDs are typically strings in OSM data, hence the str() cast from the old code
                start_str = str(row.vertex_start_id)
                end_str = str(row.vertex_end_id)

                if start_str in nodes_set and end_str in nodes_set:
                    # Assign the entire way object (tuple) to the edge ID
                    edge_to_way[row.edge_id] = way

        # 3. Build features using the lookup
        feature_list = []
        missing_count = 0

        # Iterate in the correct sequential order (0 to N-1)
        for edge_id in sorted(self.edge_as_node_map, key=lambda k: self.edge_as_node_map[k]):
            # The key is the physical edge ID. Check if it was successfully mapped to a way.
            if edge_id in edge_to_way:
                way = edge_to_way[edge_id]
                oneway = 1 if way.oneway else 0
                type_encoded = [1 if getattr(way, 'road_type', None) == rt else 0 for rt in road_types]

                feature_vector = type_encoded + [oneway]
                feature_list.append(feature_vector)
            else:
                # Default features if no way found (this is where 21312 falls)
                feature_list.append([0] * (len(road_types) + 1))
                missing_count += 1

        print(f"Warning: {missing_count} edges are missing metadata out of {len(self.edge_as_node_map)}")

        final_features = torch.tensor(feature_list, dtype=torch.float)

        # The number of static features (F) is now len(road_types) + 1
        N_nodes = final_features.shape[0]
        F_features = final_features.shape[1]

        # Log the size again to show F has increased
        print(f"(Static Feature Vector Size: N={N_nodes}, F={F_features})")

        return final_features

    def _build_line_graph_edge_index(self) -> torch.Tensor:
        """
        Build line graph edge connectivity based on shared vertices (Original Logic).
        Directed edge A -> edge B exists if A's end vertex = B's start vertex.
        The adjacency data (_adj_data) is assumed to contain 'edge_id', 'vertex_start_id', 'vertex_end_id'.
        """
        # Directed: A -> B only if A ends where B starts
        vertex_start_to_edges = defaultdict(list)  # Edges that START at this vertex
        edge_to_vertex_end = {}  # Where each edge ENDS

        # Note: We need to filter self._adj_data to only include edges in self.edge_as_node_map
        valid_adj_df = self._adj_data[
            self._adj_data["edge_id"].isin(self.edge_as_node_map.keys())
        ]

        # 1. Map vertices to edges
        for _, row in valid_adj_df.iterrows():
            edge_id = row["edge_id"]
            # Edges starting from this vertex
            vertex_start_to_edges[row["vertex_start_id"]].append(edge_id)
            # Store where this edge ends
            edge_to_vertex_end[edge_id] = row["vertex_end_id"]

        line_edges = []

        # 2. Find connections (Line Graph Edges)
        for edge_from, vertex_end in edge_to_vertex_end.items():
            # Find all edges that start where edge_from ends
            for edge_to in vertex_start_to_edges.get(vertex_end, []):
                if edge_from != edge_to:  # No self-loops
                    n1 = self.edge_as_node_map[edge_from]
                    n2 = self.edge_as_node_map[edge_to]
                    line_edges.append((n1, n2))

        if not line_edges:
            return torch.empty((2, 0), dtype=torch.long)

        # Note: If your GNN expects undirected graph, you may need to add reversed connections.
        # The SequenceDataset/GAT handles this well, so we keep it directed here.
        return torch.tensor(list(zip(*line_edges)), dtype=torch.long)

    # --- Public access to components ---
    def get_static_components(self):
        return {
            'x': self.x,
            'edge_index': self.edge_index,
            'edge_ids': self.edge_ids,
            'edge_as_node_map': self.edge_as_node_map
        }