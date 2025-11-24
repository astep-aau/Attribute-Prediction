from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.temporal_dataset_builder import TemporalDatasetBuilder
from pathlib import Path

data_dir = Path("data")
edge_files = [
    str(data_dir / "edge_data_day3.csv"),
    str(data_dir / "edge_data_day4.csv"),
    str(data_dir / "edge_data_day5.csv"),
    str(data_dir / "edge_data_day6.csv"),
    str(data_dir / "edge_data_day7.csv")
]

edge_data = ",".join(edge_files)
edge_connection = str(data_dir / "edge_connections.csv")
osm = str(data_dir / "osm_roads_output.json")


fileloader = FileLoader(edge_data, edge_connection, osm)

graph_dataset = GraphDatasetBuilder(fileloader)

temp_data_builder = TemporalDatasetBuilder(graph_dataset, 10)


print(temp_data_builder[0])
