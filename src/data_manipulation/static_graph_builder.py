# src/data_manipulation/static_graph_builder.py

import torch
import pandas as pd
from typing import Dict, Tuple, List
from collections import defaultdict
from src.data_manipulation.file_loader import FileLoader


class StaticGraphBuilder:
    def __init__(self, loader: FileLoader, global_edge_ids: List[int]):
        """
        Builds the static components of the line graph (nodes=edges) and their features.

        Args:
            loader: A FileLoader instance providing access to adjacency and metadata.
            global_edge_ids: A sorted list of all unique edge IDs defining the graph nodes.
        """
        self._adj_data = loader.get_adjacency()  # Edge connections (Original Graph Edges)
        self._meta_data = loader.get_meta_data()  # OSM road features (Way IDs as index)
        self._global_edge_ids = global_edge_ids

        # 1. Map physical edge IDs to sequential node IDs (0 to N-1)
        self.edge_as_node_map, self.edge_ids = self._create_edge_as_node_map()

        # 2. Build Static Node Features (x)
        self.x = self._build_node_features()

        # 3. Build Adjacency Matrix (edge_index) for the Line Graph
        self.edge_index = self._build_line_graph_edge_index()

    def _create_edge_as_node_map(self) -> Tuple[Dict[int, int], torch.Tensor]:
        """
        Creates a mapping from the physical road ID (int) to the graph's sequential node ID (0 to N-1).
        The resulting tensor holds the physical edge IDs, ordered by their new node index.
        """
        sorted_edge_ids = sorted(self._global_edge_ids)
        edge_as_node_map = {edge_id: i for i, edge_id in enumerate(sorted_edge_ids)}
        node_ids_tensor = torch.tensor(sorted_edge_ids, dtype=torch.int)

        return edge_as_node_map, node_ids_tensor

    def _build_node_features(self) -> torch.Tensor:
        """
        Generates the static feature tensor 'x' (N, F_static) for all nodes
        by mapping OSM metadata to the edges.
        """
        road_types = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                      "residential", "motorway_link", "trunk_link", "primary_link", "secondary_link",
                      "tertiary_link", "living_street", "service", "pedestrian", "track", "bus_guideway",
                      "escape", "raceway", "road", "busway"]

        # Filter adjacency data to only include the global set of edges (nodes)
        filtered_adj_df = self._adj_data[
            self._adj_data["edge_id"].isin(self.edge_as_node_map.keys())
        ]

        # 1. Build topological lookup (Edge ID -> OSM Way object/tuple)
        edge_to_way = {}
        for way in self._meta_data.itertuples():
            if not hasattr(way, 'nodes'):
                continue
            nodes_set = set(way.nodes)

            # Check which edges belong to this way
            for row in filtered_adj_df.itertuples():
                start_str = str(row.vertex_start_id)
                end_str = str(row.vertex_end_id)

                if start_str in nodes_set and end_str in nodes_set:
                    edge_to_way[row.edge_id] = way

        # 2. Build features in sequential node order (0 to N-1)
        feature_list = []
        missing_count = 0

        # Iterate in the correct sequential order
        for edge_id in sorted(self.edge_as_node_map, key=lambda k: self.edge_as_node_map[k]):
            if edge_id in edge_to_way:
                way = edge_to_way[edge_id]
                oneway = 1 if getattr(way, 'oneway', False) else 0  # Use getattr for safety
                type_encoded = [1 if getattr(way, 'road_type', None) == rt else 0 for rt in road_types]

                feature_vector = type_encoded + [oneway]
                feature_list.append(feature_vector)
            else:
                # Default features if no way found (all zeros)
                feature_list.append([0] * (len(road_types) + 1))
                missing_count += 1

        if missing_count > 0:
            print(
                f"StaticGraphBuilder Warning: {missing_count} edges are missing metadata out of {len(self.edge_as_node_map)}")

        final_features = torch.tensor(feature_list, dtype=torch.float)

        return final_features

    def _build_line_graph_edge_index(self) -> torch.Tensor:
        """
        Builds line graph connectivity: edge A -> edge B exists if A's end vertex = B's start vertex.
        """
        vertex_start_to_edges = defaultdict(list)
        edge_to_vertex_end = {}

        valid_adj_df = self._adj_data[
            self._adj_data["edge_id"].isin(self.edge_as_node_map.keys())
        ]

        # 1. Map vertices to edges
        for _, row in valid_adj_df.iterrows():
            edge_id = row["edge_id"]
            vertex_start_to_edges[row["vertex_start_id"]].append(edge_id)
            edge_to_vertex_end[edge_id] = row["vertex_end_id"]

        line_edges = []

        # 2. Find connections (Line Graph Edges)
        for edge_from, vertex_end in edge_to_vertex_end.items():
            for edge_to in vertex_start_to_edges.get(vertex_end, []):
                if edge_from != edge_to:
                    n1 = self.edge_as_node_map[edge_from]
                    n2 = self.edge_as_node_map[edge_to]
                    line_edges.append((n1, n2))

        if not line_edges:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(list(zip(*line_edges)), dtype=torch.long)

    def get_static_components(self):
        """
        Returns a dictionary of all static graph components ready to be passed
        to the GraphDatasetBuilder instances.
        """
        return {
            'x': self.x,
            'edge_index': self.edge_index,
            'edge_ids': self.edge_ids,
            'edge_as_node_map': self.edge_as_node_map
        }