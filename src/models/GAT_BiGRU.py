import torch.nn as nn
from torch_geometric.nn import GATConv
from src.models.config import *

class GAT_BiGRU_Imputer(nn.Module):
    def __init__(self, in_feat_static, in_feat_dynamic, gnn_hidden, gru_hidden, out_dim=1, heads=2, dropout=0.3, num_gnn_layers=1):
        super(GAT_BiGRU_Imputer, self).__init__()

        # 1. GAT (Spatial)
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout
        total_in_features = in_feat_static + in_feat_dynamic

        self.gat_layers = nn.ModuleList()

        in_dim = total_in_features

        for i in range(num_gnn_layers):
            current_heads = heads if i < num_gnn_layers - 1 else 1  # Last layer should output single head for concatenation
            current_out_dim = gnn_hidden

            self.gat_layers.append(
                GATConv(in_channels=in_dim,
                        out_channels=current_out_dim,
                        heads=current_heads,
                        dropout=dropout,
                        concat=True,  # Always concatenate for intermediate layers
                        add_self_loops=False)
            )

            # The input dimension for the next layer is the output dimension of the current layer
            in_dim = current_out_dim * current_heads

        # GAT output dimension = hidden * heads
        self.gat_out_dim = gnn_hidden * 1

        # 2. BiGRU (Temporal)
        self.bigru = nn.GRU(input_size=self.gat_out_dim,
                            hidden_size=gru_hidden,
                            num_layers=GRU_LAYERS,
                            batch_first=True,
                            bidirectional=True)

        # 3. Imputation Head (Projection)
        self.head = nn.Linear(gru_hidden * 2, out_dim)

    def forward(self, x_dynamic, x_static, edge_index, time_features):

        # 1. Get initial dimensions (x_dynamic is B, T, N, F)
        batch_size, seq_len, num_nodes, num_features = x_dynamic.shape

        if edge_index.dim() == 3:
            flat_edge_index_list = []

            # Ensure edge_index is on the same device
            if edge_index.device != x_dynamic.device:
                edge_index = edge_index.to(x_dynamic.device)

            for i in range(batch_size):
                # Shift all node indices in the current graph by (i * num_nodes)
                shifted_edges = edge_index[i] + i * num_nodes
                flat_edge_index_list.append(shifted_edges)

            # Concatenate all (2, E) tensors into one (2, B*E) tensor
            edge_index = torch.cat(flat_edge_index_list, dim=1)
        # ---------------------------------------------------------------------------

        all_time_outputs = []


        num_features_static = x_static.shape[-1]
        x_static = x_static.view(-1, num_features_static)
        # ----------------------------------------------

        # Loop over Time (T)
        for t in range(seq_len):

            # --- 1. Prepare Features for GNN Input ---

            # a. Dynamic (Travel Time)
            xt_dyn = x_dynamic[:, t, :, 0].unsqueeze(-1)
            # b. Temporal (Time of Day/Week)
            xt_time = time_features[:, t, :].unsqueeze(1)
            xt_time_expanded = xt_time.expand(-1, num_nodes, -1)
            # c. Static (Road Type/Oneway)
            xt_stat = x_static.reshape(batch_size, num_nodes, -1)

            # Combine: (B, N, Total_Features)
            xt_combined = torch.cat([xt_dyn, xt_stat, xt_time_expanded], dim=2)

            # Reshape to GNN Input: (B*N, F)
            x_b_input = xt_combined.reshape(batch_size * num_nodes, -1)


            # --- 2. Graph Attention Layers ---
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

        # 3. Stack over Time: (T, B, N, GAT_OUT_DIM)
        gnn_output_tensor = torch.stack(all_time_outputs, dim=0)

        # 4. Prepare for GRU: Reshape to (B*N, T, GAT_OUT_DIM)
        gru_in = gnn_output_tensor.permute(1, 2, 0, 3).reshape(batch_size * num_nodes, seq_len, -1)

        # 5. BiGRU
        gru_out, _ = self.bigru(gru_in)  # (B*N, T, Hidden*2)

        # 6. Imputation Head
        prediction = self.head(gru_out)  # (B*N, T, 1)

        # 7. Reshape and Transpose to match the expected output format (B, T, N)
        prediction = prediction.reshape(batch_size, num_nodes, seq_len)
        final_output = prediction.permute(0, 2, 1)

        return final_output