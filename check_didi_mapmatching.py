import pandas as pd
import numpy as np
import ast

# Check if Didi data is already map-matched
print("=== Checking Didi Chengdu Data Structure ===\n")

# Load edge features
edge_features = pd.read_csv('src/TrainingData/ExternalData/didi_chengdu/edge_features.csv')
print(f"Total road segments in network: {len(edge_features)}")
print(f"Road ID range: {edge_features['road_id'].min()} to {edge_features['road_id'].max()}\n")

# Sample a trajectory
daily_data = pd.read_csv('src/TrainingData/ExternalData/didi_chengdu/20161010.csv', nrows=10)
print("Sample trajectory paths (first 3):")
for idx, row in daily_data.head(3).iterrows():
    path = ast.literal_eval(row['path'])
    print(f"  Order {idx}: {len(path)} road segments - {path[:5]}... (showing first 5)")

# Check if path road_ids exist in edge_features
print("\nVerifying path road_ids are in edge_features:")
sample_path = ast.literal_eval(daily_data.iloc[0]['path'])
valid_roads = edge_features['road_id'].values
valid_count = sum(1 for road_id in sample_path if road_id in valid_roads)
print(f"  Sample path has {len(sample_path)} segments, {valid_count} found in edge_features ({100*valid_count/len(sample_path):.1f}%)")

# Check line graph structure
line_graph = np.load('src/TrainingData/ExternalData/didi_chengdu/line_graph_edge_idx.npy')
unique_nodes = np.unique(line_graph)
print(f"\nLine graph has {line_graph.shape[1]} edges connecting {len(unique_nodes)} unique nodes")
print(f"Node ID range: {unique_nodes.min()} to {unique_nodes.max()}")

print("\n=== CONCLUSION ===")
print("✓ Data appears to be ALREADY MAP-MATCHED to road segments")
print("✓ Trajectories contain road_id sequences (not GPS coordinates)")
print("✗ Missing: Time-slot aggregated speeds per road segment (need to aggregate)")
