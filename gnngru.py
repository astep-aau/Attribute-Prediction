import torch.nn as nn
from gnn import GraphSAGE
from gru import Gru

class GnnGru(nn.Module):
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
       super(GnnGru, self).__init__()

       self.Gnn = GraphSAGE(
               in_dim= in_dim,
               out_dim = _,
               num_layers = gnn_num_layers,
               hidden_dim = gnn_hidden_dim,
               dropout = gnn_dropout,
               aggregation_method = gnn_agg_method)

       self.Gru = Gru(
               in_dim = _,
               out_dim = out_dim,
               num_layers = gru_num_layers,
               hidden_dim = gru_hidden_dim,
               dropout = gru_dropout)

    def forward(self, node_features, edge_index):
        x = self.Gnn.forward(node_features, edge_index)
        self.Gru.forward(x)

