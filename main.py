from src.sampling.data_generation.spatial_data import generate_graph
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

print("hello")
data = generate_graph(100, 10, 50, 15)

#x = torch.randn(8, 32)  # Node features of shape [num_nodes, num_features]
#y = torch.randint(0, 4, (8, ))  # Node labels of shape [num_nodes]
#edge_index = torch.tensor([
#    [2, 3, 3, 4, 5, 6, 7],
#    [0, 0, 1, 1, 2, 3, 4]],
#)

#data = Data(x=x, y=y, edge_index = edge_index)

loader = NeighborLoader(
    data,
    input_nodes= torch.tensor([0,10,20,30,40,50,60,70,80,90]),
    num_neighbors=[2,2,2],
    batch_size=2,
    replace=False
)

for batch in loader:
    print(batch.e_id)