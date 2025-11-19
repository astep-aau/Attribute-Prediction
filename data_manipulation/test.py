# test_graph_builder.py
from file_loader import FileLoader
from graph_dataset_builder import GraphDatasetBuilder
import torch

print("building loader")
loader = FileLoader("C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day3.csv,C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day4.csv,C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day5.csv,C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day6.csv,C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day7.csv", "C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_connections.csv", "C:/Users/mikke/source/repos/Attribute-Prediction/data/osm_roads_output.json")

print("building the builder")
# Build the graph dataset (line graph: edges -> nodes)
builder = GraphDatasetBuilder(loader, timestep=0)

# Get PyTorch Geometric Data object
data = builder.get_data()

# -----------------------
# Print some info
# -----------------------
print("=== Graph Info ===")
print(f"Number of nodes (edges in original graph): {data.num_nodes}")
print(f"Number of edges in line graph: {data.num_edges}")
print(f"Node feature shape: {data.x.shape}")
print(f"Target (y) shape: {data.y.shape}")

# Print first 5 node features and corresponding travel times
print("\n=== Sample node features (first 5 nodes) ===")
print(data.x[:5])
print("\n=== Sample targets (first 5 nodes) ===")
print(data.y[:5])

# Print first 10 edges in edge_index
print("\n=== Sample edges in line graph (first 10) ===")
print(data.edge_index[:, :10])
