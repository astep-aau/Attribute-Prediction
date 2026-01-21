import pandas as pd
import numpy as np

# Check the structure of didi_chengdu dataset
print("=== Edge Features ===")
edge_features = pd.read_csv('src/TrainingData/ExternalData/didi_chengdu/edge_features.csv')
print(f"Shape: {edge_features.shape}")
print(f"Columns: {edge_features.columns.tolist()}")
print(f"First 3 rows:\n{edge_features.head(3)}")

print("\n=== Line Graph Edge Index ===")
line_graph = np.load('src/TrainingData/ExternalData/didi_chengdu/line_graph_edge_idx.npy')
print(f"Shape: {line_graph.shape}")
print(f"First 10 edges (source, target):\n{line_graph[:, :10].T}")

print("\n=== Daily CSV Structure (sample) ===")
daily_data = pd.read_csv('src/TrainingData/ExternalData/didi_chengdu/20161010.csv', nrows=5)
print(f"Shape: {daily_data.shape}")
print(f"First 10 columns: {daily_data.columns.tolist()[:10]}")
print(f"Sample data:\n{daily_data.iloc[:3, :5]}")
