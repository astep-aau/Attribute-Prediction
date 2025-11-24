import torch
from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder

class TemporalDatasetBuilder():
    def __init__(self, graph_dataset_builder, sequence_length):
        """
        Args:
            graph_dataset_builder: GraphDatasetBuilder instance
            sequence_length: How many past timesteps to use
        """
        # Store these
        self.sequence_length = sequence_length

        # Load the data
        self.graph_builder = graph_dataset_builder

        # Get static graph structure
        self.static_features = self.graph_builder.x       # [num_nodes, static_dim]

    def __len__(self):
        """Total number of sequences that can be created"""
        # Need sequence_length timesteps for input, so max valid idx is total - sequence_length
        return len(self.graph_builder._travel_data) - self.sequence_length

    def __getitem__(self, idx):
        """
        Get one training sample

        Returns:
            graph_sequence: [seq_len, num_nodes, in_dim]
        """
        timestep_features_list = []
        for t in range(idx, idx + self.sequence_length):
            travel_times = self.graph_builder._build_target_tensor(t)

            travel_times = travel_times.unsqueeze(1) # [num_nodes, 1]

            # Concat static features
            combined = torch.cat([travel_times, self.static_features], dim=1) # [num_nodes, 23] (1 travel_time + 22 static features)

            timestep_features_list.append(combined)

        # Stack into sequence
        graph_sequence = torch.stack(timestep_features_list, dim=0) # [sequence_length, num_nodes, 23]

        return graph_sequence
