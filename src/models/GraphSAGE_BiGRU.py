import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from src.models.config import *
from src.models.logging_utils import logger


class GraphSAGE_BiGRU_Imputer(nn.Module):
    # --- UPDATED CONSTRUCTOR SIGNATURE ---
    def __init__(self, in_feat, gnn_hidden, gru_hidden, out_dim=1, heads=2, dropout=0.3, num_gnn_layers=1):
        super(GraphSAGE_BiGRU_Imputer, self).__init__()

        # --- GNN Output Dimension ---
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout
        self.sage_output_dim = gnn_hidden

        # 1. GraphSAGE Layers
        self.sage_layers = nn.ModuleList()
        self.dropout_layer = nn.Dropout(p=dropout)

        # Total input features is simply 'in_feat' (e.g., 32)
        in_dim = in_feat

        for i in range(num_gnn_layers):
            current_out_dim = gnn_hidden

            self.sage_layers.append(
                SAGEConv(in_channels=in_dim,
                         out_channels=current_out_dim)
            )

            in_dim = current_out_dim  # Output becomes input

        self.gru_input_size = self.sage_output_dim

        # 2. BiGRU (Temporal)
        self.bigru = nn.GRU(input_size=self.sage_output_dim,
                            hidden_size=gru_hidden,
                            num_layers=GRU_LAYERS,
                            batch_first=True,
                            bidirectional=True)

        # 3. Imputation Head (Projection)
        self.head = nn.Linear(gru_hidden * 2, out_dim)

    # --- UPDATED FORWARD SIGNATURE ---
    def forward(self, x_combined, edge_index):

        # x_combined is (B, T, N, F_feat)
        batch_size, seq_len, num_nodes, num_features = x_combined.shape

        # 1. Perform the shift and concatenation of edge_index for batched graph processing
        flat_edge_index_list = []

        if edge_index.device != x_combined.device:
            edge_index = edge_index.to(x_combined.device)

        for i in range(batch_size):
            shifted_edges = edge_index[i] + i * num_nodes
            flat_edge_index_list.append(shifted_edges)

        edge_index = torch.cat(flat_edge_index_list, dim=1)

        all_time_outputs = []

        # Loop over Time (T)
        for t in range(seq_len):
            # 2. Get GNN Input: (B, N, F_feat) -> (B*N, F_feat)
            xt_combined = x_combined[:, t, :, :]
            x_b_input = xt_combined.reshape(batch_size * num_nodes, -1)

            # 3. GraphSAGE Layers
            h_t = x_b_input
            for i, layer in enumerate(self.sage_layers):
                h_t = layer(h_t, edge_index)

                # Apply ReLU and Dropout to all intermediate layers
                if i < self.num_gnn_layers - 1:
                    h_t = torch.relu(h_t)
                    h_t = self.dropout_layer(h_t)

            # Reshape GNN output back to (B, N, SAGE_OUT_DIM)
            h_t = h_t.reshape(batch_size, num_nodes, -1)
            all_time_outputs.append(h_t)

        # 4. Stack over Time: (T, B, N, SAGE_OUT_DIM)
        gnn_output_tensor = torch.stack(all_time_outputs, dim=0)

        # 5. Prepare for GRU: Reshape to (B*N, T, SAGE_OUT_DIM)
        gru_in = gnn_output_tensor.permute(1, 2, 0, 3).reshape(batch_size * num_nodes, seq_len, -1)

        # 6. BiGRU
        gru_out, _ = self.bigru(gru_in)  # (B*N, T, Hidden*2)

        # 7. Imputation Head
        prediction = self.head(gru_out)  # (B*N, T, 1)

        # 8. Reshape and Transpose to match the expected output format (B, T, N)
        prediction = prediction.reshape(batch_size, num_nodes, seq_len)
        final_output = prediction.permute(0, 2, 1)

        return final_output