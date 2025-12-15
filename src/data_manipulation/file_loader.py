from src.data_manipulation.data_loader import DataLoader
import pandas as pd

class FileLoader(DataLoader):
    def __init__(self, edge_data_path: str, edge_connections_path, meta_data_path):
        self.edge_data_path = edge_data_path
        self.edge_connections_path = edge_connections_path
        self.meta_data_path = meta_data_path

    def get_travel_data(self):
        df = pd.read_csv(self.edge_data_path)
        all_edges = set(df.columns)
        all_edges.discard("time_slot")

        # Sort edges (only used to ensure column order, still relevant)
        edges_sorted = sorted(all_edges, key=lambda x: int(x.split("_")[0].replace("edge", "")))
        master_cols = ["time_slot"] + edges_sorted

        # --- REMOVED: base_date and day offset logic ---
        base_date = pd.to_datetime("2025-11-26 00:00:00")

        # Fill missing edges with -1 (Necessary if not all 5 days have the same edge set)
        missing_cols = {col: -1 for col in master_cols if col not in df.columns}
        if missing_cols:
            df = pd.concat([df, pd.DataFrame(missing_cols, index=df.index)], axis=1)
        df = df[master_cols]

        # 1. Calculate Time Offset from the [hours;minutes] data
        time_parts = df["time_slot"].str.split(':', expand=True).astype(int)
        hours = time_parts[0]
        minutes = time_parts[1]

        time_offset = pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')

        # 2. Create the final Timestamp column
        df["Timestamp"] = base_date + time_offset

        return df

    def get_meta_data(self):
        # Transpose so edge IDs become the index and columns are oneway, road_type, nodes
        return pd.read_json(self.meta_data_path).T

    def get_adjacency(self):
        return pd.read_csv(self.edge_connections_path)
