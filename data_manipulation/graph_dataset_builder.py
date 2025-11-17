from data_loader import DataLoader
import torch
from torch_geometric.data import Data

class GraphDatasetBuilder:
    def __init__(self, loader : DataLoader):
        self._meta_data_df = loader.getMetaData()
        self._adjacency_df = loader.getAdjacency()
        self._travel_data = loader.getMetaData()
        self._id_map = self._create_id_map()
        self.edge_index = self.buildEdgeList()

    def buildEdgeList(self):
        return torch.tensor([
                [self._id_map[x] for x in self._adjacency_df["vertex_start_id"].tolist()], 
                [self._id_map[x] for x in self._adjacency_df["vertex_end_id"].tolist()],
            ], dtype=torch.long)

    def _create_id_map(self):
        return {old_id: new_id for new_id, old_id in enumerate(self._meta_data_df["node_id"])}
