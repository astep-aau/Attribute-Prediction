# src/data_manipulation/file_loader.py

import pandas as pd
import os
from typing import List, Dict


class FileLoader:
    # --- MODIFIED: Added master_cols argument ---
    def __init__(self, edge_data_path: str, edge_connections_path: str, meta_data_path: str, master_cols: List[str]):
        """
        Initializes the FileLoader with paths and the global master column list.
        """
        self.edge_data_path = edge_data_path
        self.edge_connections_path = edge_connections_path
        self.meta_data_path = meta_data_path
        self.master_cols = master_cols  # List of all columns required across all days

    def get_travel_data(self) -> pd.DataFrame:
        """
        Loads the daily travel data and ensures it conforms to the global master_cols set.
        Missing columns are padded with -1.0.
        """
        df = pd.read_csv(self.edge_data_path)

        # --- CRITICAL FIX FOR CONSISTENT N (NODE COUNT) ---

        # 1. Identify columns present in master_cols but missing in the daily file
        missing_cols = [col for col in self.master_cols if col not in df.columns]

        # 2. Add missing columns, filling them with the missing data sentinel (-1.0)
        if missing_cols:
            missing_data = pd.DataFrame({col: -1.0 for col in missing_cols}, index=df.index)
            df = pd.concat([df, missing_data], axis=1)

        # 3. Ensure the final DataFrame has the exact global column order and set
        df = df[self.master_cols]
        # --- END CRITICAL FIX ---

        # ... (rest of time calculation logic should remain the same) ...
        base_date = pd.to_datetime("2025-11-26 00:00:00")
        time_parts = df["time_slot"].str.split(':', expand=True).astype(int)
        time_offset = pd.to_timedelta(time_parts[0], unit='h') + pd.to_timedelta(time_parts[1], unit='m')
        df["Timestamp"] = base_date + time_offset

        return df

    # --- get_adjacency ---
    def get_adjacency(self) -> pd.DataFrame:
        # Assuming your adjacency file has 'edge_id', 'vertex_start_id', 'vertex_end_id'
        df = pd.read_csv(self.edge_connections_path)
        return df

    # --- get_meta_data ---
    def get_meta_data(self) -> pd.DataFrame:
        """
        Loads the OSM metadata, transposes it, and ensures the Edge ID index is an integer.
        """
        df = pd.read_json(self.meta_data_path).T

        df.index = df.index.astype(int)

        return df