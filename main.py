from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.sampling.neighbor_sampling import NeighborSampling
from pathlib import Path

data_dir = Path("data")
edge_files = [str(data_dir / "edge_data_day3.csv")]

edge_data = ",".join(edge_files)
edge_connection = str(data_dir / "edge_connections.csv")
osm = str(data_dir / "osm_roads_output.json")

fileloader = FileLoader(edge_data, edge_connection, osm)
graph_builder = GraphDatasetBuilder(fileloader)

data = graph_builder.get_data()

print(data)

sampledData = NeighborSampling(data)

print(sampledData)

for batch in sampledData:
    print(batch)