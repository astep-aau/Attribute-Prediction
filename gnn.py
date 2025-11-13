import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        num_layers,
        hidden_dim,
        dropout,
        aggregation_method = "mean"):
            super(GraphSAGE, self).__init__()

            self.convs = nn.ModuleList()
            self.num_layers = num_layers
            self.dropout = dropout

            # Input layer
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggregation_method))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggregation_method))

            # Output layer
            self.convs.append(SAGEConv(hidden_dim, out_dim, aggregation_method))

    def forward(self, node_feature, edge_index):
        """
            node_feature = matrix [num_nodes, in_dim]
            edge_index = COO format [2, num_edges]
        """
        for i, conv in enumerate(self.convs):
            node_feature = conv(node_feature, edge_index)

            if i != self.num_layers - 1: # no activation function on last layer
                node_feature = F.relu(node_feature)
                node_feature = F.dropout(node_feature, p=self.dropout, training=self.training)
        return node_feature
