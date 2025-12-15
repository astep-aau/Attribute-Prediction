import torch.nn as nn
from torch_geometric.nn import GATConv
from src.models.config import *
import torch  # Import torch explicitly


class GAT_BiGRU_Imputer(nn.Module):
    # --- UPDATED CONSTRUCTOR SIGNATURE ---
    def __init__(self, in_feat, gnn_hidden, gru_hidden, out_dim=1, heads=2, dropout=0.3, num_gnn_layers=1):
        super(GAT_BiGRU_Imputer, self).__init__()

        # 1. GAT (Spatial)
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout

        # Total input features is simply 'in_feat' (e.g., 32)
        in_dim = in_feat
        self.gat_layers = nn.ModuleList()

        for i in range(num_gnn_layers):
            # Last layer should output single head for easy concatenation/GRU input
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
            # The input dimension for the next layer is the output dimension of the current layer
            in_dim = current_out_dim * current_heads

        # GAT output dimension is the output of the final layer
        # Since the last layer uses current_heads=1, the out dim is gnn_hidden * 1
        self.gat_out_dim = gnn_hidden * 1

        # 2. BiGRU (Temporal)
        self.bigru = nn.GRU(input_size=self.gat_out_dim,
                            hidden_size=gru_hidden,
                            num_layers=GRU_LAYERS,
                            batch_first=True,
                            bidirectional=True)

        # 3. Imputation Head (Projection)
        self.head = nn.Linear(gru_hidden * 2, out_dim)

    # --- UPDATED FORWARD SIGNATURE ---
    # It now only accepts the combined input (X_feat_input from trainer) and edge_index
    def forward(self, x_combined, edge_index):

        # x_combined is (B, T, N, F_feat)
        batch_size, seq_len, num_nodes, num_features = x_combined.shape

        # 1. Perform the shift and concatenation of edge_index for batched graph processing
        if edge_index.dim() == 3:
            flat_edge_index_list = []

            # Ensure edge_index is on the same device
            if edge_index.device != x_combined.device:
                edge_index = edge_index.to(x_combined.device)

            for i in range(batch_size):
                # Shift all node indices in the current graph by (i * num_nodes)
                shifted_edges = edge_index[i] + i * num_nodes
                flat_edge_index_list.append(shifted_edges)

            # Concatenate all (2, E) tensors into one (2, B*E) tensor
            edge_index = torch.cat(flat_edge_index_list, dim=1)

        all_time_outputs = []

        # --- Loop over Time (T) ---
        for t in range(seq_len):

            # 2. Get GNN Input: (B, N, F_feat) -> (B*N, F_feat)
            # Input is already combined by the SequenceDataset
            xt_combined = x_combined[:, t, :, :]  # (B, N, F_feat)
            x_b_input = xt_combined.reshape(batch_size * num_nodes, -1)  # (B*N, F_feat)

            # 3. Graph Attention Layers
            h_t = x_b_input
            for i, layer in enumerate(self.gat_layers):
                h_t = layer(h_t, edge_index)

                # Apply activation (ReLU) and Dropout to intermediate layers
                if i < self.num_gnn_layers - 1:
                    h_t = torch.relu(h_t)
                    h_t = torch.dropout(h_t, p=self.dropout_rate, train=self.training)

            # Reshape GNN output back to (B, N, GAT_OUT_DIM)
            h_t = h_t.reshape(batch_size, num_nodes, -1)
            all_time_outputs.append(h_t)

        # 4. Stack over Time: (T, B, N, GAT_OUT_DIM)
        gnn_output_tensor = torch.stack(all_time_outputs, dim=0)

        # 5. Prepare for GRU: Reshape to (B*N, T, GAT_OUT_DIM)
        # Permute (T, B, N, F) -> (B, N, T, F) -> (B*N, T, F)
        gru_in = gnn_output_tensor.permute(1, 2, 0, 3).reshape(batch_size * num_nodes, seq_len, -1)

        # 6. BiGRU
        gru_out, _ = self.bigru(gru_in)  # (B*N, T, Hidden*2)

        # 7. Imputation Head
        prediction = self.head(gru_out)  # (B*N, T, 1)

        # 8. Reshape and Transpose to match the expected output format (B, T, N)
        prediction = prediction.reshape(batch_size, num_nodes, seq_len)
        final_output = prediction.permute(0, 2, 1)  # (B, T, N)

        return final_output