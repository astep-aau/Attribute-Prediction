# src/data_manipulation/file_loader.py

import pandas as pd
from typing import List, Dict


class FileLoader:
    def __init__(self, edge_data_path: str, edge_connections_path: str, meta_data_path: str, master_cols: List[str]):
        """
        Initializes the FileLoader with file paths and the global master column list
        to ensure data consistency across daily files.
        """
        self.edge_data_path = edge_data_path
        self.edge_connections_path = edge_connections_path
        self.meta_data_path = meta_data_path
        self.master_cols = master_cols

    def get_travel_data(self) -> pd.DataFrame:
        """
        Loads the daily travel time data, ensures it conforms to the global master_cols
        (padding missing columns with the sentinel value -1.0), and computes the 'Timestamp'.
        """
        df = pd.read_csv(self.edge_data_path)

        # 1. Pad missing columns (nodes) with -1.0 to ensure consistent node count (N)
        missing_cols = [col for col in self.master_cols if col not in df.columns]

        if missing_cols:
            missing_data = pd.DataFrame({col: -1.0 for col in missing_cols}, index=df.index)
            df = pd.concat([df, missing_data], axis=1)

        # 2. Ensure the final DataFrame has the exact global column order and set
        df = df[self.master_cols]

        # 3. Calculate Timestamp
        base_date = pd.to_datetime("2025-11-26 00:00:00")
        time_parts = df["time_slot"].str.split(':', expand=True).astype(int)
        time_offset = pd.to_timedelta(time_parts[0], unit='h') + pd.to_timedelta(time_parts[1], unit='m')
        df["Timestamp"] = base_date + time_offset

        return df

    def get_adjacency(self) -> pd.DataFrame:
        """
        Loads the edge connectivity data (adjacency list).
        """
        df = pd.read_csv(self.edge_connections_path)
        return df

    def get_meta_data(self) -> pd.DataFrame:
        """
        Loads the OSM metadata, transposes it (rows are edges), and ensures the Edge ID index is an integer.
        """
        df = pd.read_json(self.meta_data_path).T
        df.index = df.index.astype(int)
        return df