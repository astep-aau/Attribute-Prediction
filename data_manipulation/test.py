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

# Check feature dimensions
expected_features = 22  # 21 road types + 1 oneway
print(f"\nExpected features per node: {expected_features}")
print(f"Actual features per node: {data.x.shape[1]}")

# Print first 5 node features and corresponding travel times
print("\n=== Sample node features (first 5 nodes) ===")
print(data.x[:5])

# Check if features are not all zeros
non_zero_features = (data.x != 0).any(dim=1).sum().item()
print(f"\nNodes with non-zero features: {non_zero_features} / {data.num_nodes}")

# Check feature statistics
print("\n=== Feature Statistics ===")
print(f"Min value: {data.x.min().item():.4f}")
print(f"Max value: {data.x.max().item():.4f}")
print(f"Mean value: {data.x.mean().item():.4f}")
print(f"Non-zero values: {(data.x != 0).sum().item()} / {data.x.numel()}")

# Check road type distribution (sum of one-hot encoding in first 21 features)
road_type_counts = data.x[:, :21].sum(dim=0)
print("\n=== Road Type Distribution (first 21 features) ===")
road_types = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
              "residential", "motorway_link", "trunk_link", "primary_link", "secondary_link",
              "tertiary_link", "living_street", "service", "pedestrian", "track", "bus_guideway",
              "escape", "raceway", "road", "busway"]
for i, road_type in enumerate(road_types):
    count = road_type_counts[i].item()
    if count > 0:
        print(f"{road_type}: {int(count)}")

# Check oneway distribution (last feature)
oneway_count = data.x[:, -1].sum().item()
print(f"\n=== Oneway Feature (last column) ===")
print(f"Oneway roads: {int(oneway_count)} / {data.num_nodes}")
print(f"Bidirectional roads: {data.num_nodes - int(oneway_count)} / {data.num_nodes}")

print("\n=== Sample targets (first 5 nodes) ===")
print(data.y[:5])

# Print first 10 edges in edge_index
print("\n=== Sample edges in line graph (first 10) ===")
print(data.edge_index[:, :10])
