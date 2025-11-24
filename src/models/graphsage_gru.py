import torch
import torch.nn as nn
from src.models.graphsage import GraphSAGEGru
from gru import Gru

class GraphSAGEGru(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            gnn_num_layers,
            gru_num_layers,
            gnn_hidden_dim,
            gru_hidden_dim,
            gnn_dropout,
            gru_dropout,
            gnn_agg_method):
       super(GraphSAGEGru, self).__init__()

       self.Gnn = GraphSAGEGru(
               in_dim= in_dim,
               out_dim = gnn_hidden_dim,  # Keep as embeddings, not predictions
               num_layers = gnn_num_layers,
               hidden_dim = gnn_hidden_dim,
               dropout = gnn_dropout,
               aggregation_method = gnn_agg_method)

       self.Gru = Gru(
               in_dim = gnn_hidden_dim,  # Matches GNN output
               out_dim = out_dim,
               num_layers = gru_num_layers,
               hidden_dim = gru_hidden_dim,
               dropout = gru_dropout)

    def forward(self, graph_sequence, edge_index, h0=None):
        """
        For road network travel time prediction.

        Args:
            graph_sequence: [seq_len, num_nodes, in_dim] - historical node features over time
            edge_index: [2, num_edges] - static road network structure (line graph)
            h0: [num_layers, num_nodes, gru_hidden_dim] - optional hidden state per node

        Returns:
            out: [num_nodes, out_dim] - travel time prediction per road segment
            hn: [num_layers, num_nodes, gru_hidden_dim] - hidden state for next sequence
        """
        seq_len, num_nodes, _ = graph_sequence.shape

        # Process each timestep with GNN to capture spatial dependencies
        spatial_temporal_features = []
        for t in range(seq_len):
            node_features = graph_sequence[t]  # [num_nodes, in_dim]

            spatial_features = self.Gnn.forward(node_features, edge_index)  # [num_nodes, gnn_hidden_dim]
            spatial_temporal_features.append(spatial_features)

        # Stack: [seq_len, num_nodes, gnn_hidden_dim]
        gnn_sequence = torch.stack(spatial_temporal_features, dim=0)

        # Transpose for GRU: [num_nodes, seq_len, gnn_hidden_dim]
        # Each road segment has its own temporal sequence
        gnn_sequence = gnn_sequence.transpose(0, 1)

        # GRU processes temporal patterns independently per road segment
        out, hn = self.Gru.forward(gnn_sequence, h0)  # out: [num_nodes, out_dim]

        return out, hn
