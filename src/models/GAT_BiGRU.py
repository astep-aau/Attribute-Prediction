import torch.nn as nn
from torch_geometric.nn import GATConv
from src.models.config import *
import torch


class GAT_BiGRU_Imputer(nn.Module):
    def __init__(self, in_feat, gnn_hidden, gru_hidden, out_dim=1, heads=2, dropout=0.3, num_gnn_layers=1):
        """
        GAT-BiGRU model for Spatio-Temporal imputation using a combined feature vector.
        """
        super(GAT_BiGRU_Imputer, self).__init__()

        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout

        in_dim = in_feat
        self.gat_layers = nn.ModuleList()

        # 1. GAT Layers (Spatial Feature Extraction)
        for i in range(num_gnn_layers):
            current_heads = heads if i < num_gnn_layers - 1 else 1
            current_out_dim = gnn_hidden

            self.gat_layers.append(
                GATConv(in_channels=in_dim,
                        out_channels=current_out_dim,
                        heads=current_heads,
                        dropout=dropout,
                        concat=True,
                        add_self_loops=False)
            )
            in_dim = current_out_dim * current_heads

        self.gat_out_dim = gnn_hidden * 1  # Output size of the final GAT layer

        # 2. BiGRU (Temporal Feature Extraction)
        self.bigru = nn.GRU(input_size=self.gat_out_dim,
                            hidden_size=gru_hidden,
                            num_layers=GRU_LAYERS,
                            batch_first=True,
                            bidirectional=True)

        # 3. Imputation Head (Projection)
        self.head = nn.Linear(gru_hidden * 2, out_dim)

    def forward(self, x_combined: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_combined: Combined input tensor (B, T, N, F_feat).
            edge_index: Adjacency list (B, 2, E) or (2, B*E).

        Returns:
            Prediction tensor (B, T, N).
        """
        batch_size, seq_len, num_nodes, _ = x_combined.shape

        # 1. Concatenate edge_index for batched graph processing
        if edge_index.dim() == 3:
            flat_edge_index_list = []
            if edge_index.device != x_combined.device:
                edge_index = edge_index.to(x_combined.device)

            for i in range(batch_size):
                shifted_edges = edge_index[i] + i * num_nodes
                flat_edge_index_list.append(shifted_edges)

            edge_index = torch.cat(flat_edge_index_list, dim=1)

        all_time_outputs = []

        # 2. Loop over Time (T) and apply GAT
        for t in range(seq_len):
            xt_combined = x_combined[:, t, :, :]  # (B, N, F_feat)
            x_b_input = xt_combined.reshape(batch_size * num_nodes, -1)  # (B*N, F_feat)

            h_t = x_b_input
            for i, layer in enumerate(self.gat_layers):
                h_t = layer(h_t, edge_index)

                if i < self.num_gnn_layers - 1:
                    h_t = torch.relu(h_t)
                    h_t = torch.dropout(h_t, p=self.dropout_rate, train=self.training)

            h_t = h_t.reshape(batch_size, num_nodes, -1)
            all_time_outputs.append(h_t)

        # 3. Stack GNN outputs
        gnn_output_tensor = torch.stack(all_time_outputs, dim=0)  # (T, B, N, GAT_OUT_DIM)

        # 4. Prepare for BiGRU: (B, N, T, F) -> (B*N, T, F)
        gru_in = gnn_output_tensor.permute(1, 2, 0, 3).reshape(batch_size * num_nodes, seq_len, -1)

        # 5. BiGRU
        gru_out, _ = self.bigru(gru_in)

        # 6. Imputation Head
        prediction = self.head(gru_out)

        # 7. Reshape to final output format (B, T, N)
        prediction = prediction.reshape(batch_size, num_nodes, seq_len)
        final_output = prediction.permute(0, 2, 1)

        final_output = torch.relu(final_output)

        return final_output