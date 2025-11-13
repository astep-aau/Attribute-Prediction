import torch.nn as nn

class gru(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            hidden_dim,
            num_layers,
            dropout):
        super(gru, self).__init__()
        
        # HyperParams
        self.hidden_dim = hidden_dim
        self.num_layer = num_layers
        self.dropout = dropout

        # GRU layer
        self.gru = nn.GRU(
                input_size = in_dim,
                hidden_size = hidden_dim,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout if num_layers > 1 else 0          
                )
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, h0=None):
        out, hn = self.gru(x, h0)
        out = out[:, -1, :] # [Batch_size, out_dim]
        out = self.fc(out)
        return out, hn
