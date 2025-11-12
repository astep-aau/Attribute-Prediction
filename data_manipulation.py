import os, json
import numpy as np
import pandas as pd
import torch

class DataManipulation():
    def __init__(self, path_to_data, min_event_threshold, sequance_length):
        base_path = os.path.dirname(__file__)
        self._path_to_data = os.path.join(base_path, path_to_data)
        self._min_event_threshold = min_event_threshold
        self._sequance_length = sequance_length

        self._trainX = None
        self._trainY = None

        self.df_data = None
        with open(self._path_to_data, "r") as f:
            raw_data = json.load(f)

        self.data = self._filter_roads_by_length(raw_data)

        self._load_data_into_pd()

    def get_trainX(self):
        if self._trainX is None:
            self._create_sequance(self.df_data, self._sequance_length)
        return self._trainX

    def get_trainY(self):
        if self._trainY is None:
            self._create_sequance(self.df_data, self._sequance_length)
        return self._trainY
    def _load_data_into_pd(self):
        csv_name = "data/min_event_threshold_" + str(self._min_event_threshold) + "_lstm_data.csv"
        if os.path.exists(csv_name):
            self.df_data = pd.read_csv(csv_name)
            return

        truncation_length = self._find_truncation_length()

        data_for_df = {}
        all_road_ids = sorted(self.data.keys())

        target_key = "time to traverse (s)"
        for road_id in all_road_ids:
            road_data_dict = self.data[road_id]

            full_series = [
                road_data_dict.get("traversals", {}).get(str(i), {}).get(target_key, np.nan)  for i in range(truncation_length)
            ]

            data_for_df[road_id] = full_series

        df = pd.DataFrame(data_for_df)

        nan_count_before = df.isna().sum().sum()

        total_data_points = df.size # df.size is equivalent to rows * columns

        if total_data_points > 0:
            missing_percentage = (nan_count_before / total_data_points) * 100
            print(f"--- Data Quality Report ---")
            print(f"DataFrame shape before interpolation: {df.shape}")
            print(f"Total data points: {total_data_points}")
            print(f"Missing data points (NaNs) to be filled: {nan_count_before}")
            print(f"Percentage of data being interpolated: {missing_percentage:.2f}%")

        self.df_data = df.interpolate(method='linear', limit_direction='both')
        self.df_data.to_csv(csv_name, index=False)

    def _filter_roads_by_length(self, raw_data):
        filtered_data = {}
        for road_id, road_data in raw_data.items():
            traversals_data = road_data.get("traversals")
            if not traversals_data:
                continue # Skip if no traversals key

            event_count = len([k for k in traversals_data.keys() if k.isdigit()])

            # This is the core filtering logic
            if event_count >= self._min_event_threshold:
                filtered_data[road_id] = road_data

        print(f"Original number of roads: {len(raw_data)}")
        print(f"Number of roads after filtering (threshold >= {self._min_event_threshold}): {len(filtered_data)}")
        return filtered_data

    def _create_sequance(self, df, seq_len):
        x = []
        y = []
        for column_name in df:
            road_series = df[column_name].values
            if len(road_series) <= seq_len:
                continue
            for i in range(seq_len, len(road_series)):
                x.append(road_series[i-seq_len:i])
                y.append(road_series[i])
        np_x = np.array(x)
        np_y = np.array(y)

        # (batch_size, sequence_length, input_size)
        self._trainX = torch.tensor(np_x, dtype=torch.float32)[:, :, None]
        self._trainY = torch.tensor(np_y, dtype=torch.float32)[:, None]

    def _find_truncation_length(self):
        min_of_max_timesteps = float('inf')

        for road_data in self.data.values():
            if not road_data:
                continue
            traversals_data = road_data.get("traversals")
            if not traversals_data:
                continue

            # For the current road, find its own maximum timestamp
            current_road_max_timestep = max([int(t) for t in traversals_data.keys()])

            # Check if this road's max length is smaller than the smallest we've seen so far
            if current_road_max_timestep < min_of_max_timesteps:
                min_of_max_timesteps = current_road_max_timestep

        # The total length is the max index + 1 (e.g., index 155 means 156 steps)
        # If no data was found, return 0.
        print("trancation length: " + str(min_of_max_timesteps))
        return min_of_max_timesteps
