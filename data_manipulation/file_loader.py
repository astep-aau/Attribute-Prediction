from data_loader import DataLoader
import pandas as pd
import json

class FileLoader(DataLoader):
    def __init__(self, edge_data_paths: str, edge_connections_path, meta_data_path):
        self.edge_data_paths = edge_data_paths.split(",")
        self.edge_connections_path = edge_connections_path
        self.meta_data_path = meta_data_path

    def getTravelData(self):
        dataframes = [pd.read_csv(e) for e in self.edge_data_paths]
        all_edges = set()

        for dataframe in dataframes:
            all_edges |= set(dataframe.columns)
        all_edges.discard("time_slot")

        # Sort edges
        edges_sorted = sorted(all_edges, key=lambda x: int(x.split("_")[0].replace("edge","")))
        master_cols = ["time_slot"] + edges_sorted
        new_dataframes = []
        for i, df in enumerate(dataframes):
            # Fill missing edges with -1
            missing_cols = {col: -1 for col in master_cols if col not in df.columns}
            if missing_cols:
                df = pd.concat([df, pd.DataFrame(missing_cols, index=df.index)], axis=1)
            df = df[master_cols]
            df["day"] = i + 1
            new_dataframes.append(df)

        return pd.concat(new_dataframes, ignore_index=True)

    def getMetaData(self):
        with open(self.meta_data_path, "r") as f:
            data = json.load(f)

        rows = [
            {
                "node_id": node,
                "oneway": info["oneway"],
                "road_type": info["road_type"]
            }
            for info in data.values()         # iterate over the value dicts directly
            for node in info["nodes"]         # iterate over each node in the list
        ]
        return pd.DataFrame(rows)

    def getAdjacency(self):
        return pd.read_csv(self.edge_connections_path)
