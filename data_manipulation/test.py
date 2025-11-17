from data_loader import DataLoader
from file_loader import FileLoader 

obj = FileLoader("C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day3.csv,C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day4.csv,C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day5.csv,C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day6.csv,C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_data_day7.csv", "C:/Users/mikke/source/repos/Attribute-Prediction/data/edge_connections.csv", "C:/Users/mikke/source/repos/Attribute-Prediction/data/osm_roads_output.json")

df = obj.getAdjacency()
print(f"Shape: {df.shape}")
df = obj.getMetaData()
print(f"Shape: {df.shape}")
df = obj.getTravelData()
print(f"Shape: {df.shape}")

