import torch
from torch_geometric.loader import NeighborLoader

def NeighborSampling(data):
    loader = NeighborLoader(
        data,
        input_nodes= torch.arange(data.num_nodes),
        num_neighbors=[-1,-1],
        batch_size=128,
        replace=False
    )

    return loader