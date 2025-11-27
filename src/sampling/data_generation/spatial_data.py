import torch
import random
from torch_geometric.data import Data

def generate_graph(node_amount, min_travel_time, max_travel_time, intersection_amount):
    if node_amount % 2:
        print("node_amount must be even")
        return

    #Generate random values for each node
    travel_time = torch.rand(node_amount) * (max_travel_time-min_travel_time) + min_travel_time

    #Turn it into shape [num_node, num_features] (we only have one feature, travel_time, hence the shape always becomes [node_amount, 1])
    x = travel_time.unsqueeze(-1)

    #Random edges between nodes, max 6 connections, all roads are bidrectional.

    edges = []

    for i in range(0, node_amount-3, 2):
        edges.append([i, i + 2])
        edges.append([i + 3, i + 1])

    edges.append([node_amount-2, 0])
    edges.append([1, node_amount-1])
    
    for i in range(intersection_amount):
        
        random_road1 = random.randint(0, int(node_amount/2) - 1) * 2
        random_road2 = random.randint(0, int(node_amount/2) - 1) * 2 + 1

        if random_road1 == random_road2 - 1:
            random_road1 = random_road1 - 2
            if random_road1 < 0:
                random_road1 + 1 + intersection_amount

        edges.append([random_road1, random_road2])
        edges.append([random_road1, random_road2 + 1])

        edges.append([random_road2, random_road1])
        edges.append([random_road2, random_road1 + 1])

    edge_index = torch.tensor(edges, dtype = torch.long).t().contiguous()

    data = Data(
        x = x,
        edge_index = edge_index,
    )
    return data