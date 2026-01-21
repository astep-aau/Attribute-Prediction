"""
Didi Chengdu to Harbin Format Converter
Aggregates trajectory data into time-slot based edge speeds
"""
import pandas as pd
import numpy as np
import ast
import json
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DidiToHarbinConverter:
    def __init__(self, didi_dir, output_dir, time_slot_minutes=5):
        """
        Args:
            didi_dir: Path to didi_chengdu folder
            output_dir: Path to save converted files
            time_slot_minutes: Time slot interval (default 5 minutes like Harbin)
        """
        self.didi_dir = Path(didi_dir)
        self.output_dir = Path(output_dir)
        self.time_slot_minutes = time_slot_minutes
        
        # Create output directories
        (self.output_dir / 'days').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'connections').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'osm_data').mkdir(parents=True, exist_ok=True)
        
        # Load edge features
        self.edge_features = pd.read_csv(self.didi_dir / 'edge_features.csv')
        self.num_edges = len(self.edge_features)
        logger.info(f"Loaded {self.num_edges} road segments")
        
    def aggregate_trajectories_to_timeslots(self, date_file):
        """
        Aggregate trajectories from a daily file into time-slot based speeds.
        Returns DataFrame with columns: time_slot, edge0_traversal_time_sec, edge1_traversal_time_sec, ...
        """
        logger.info(f"Processing {date_file.name}...")
        
        # Read trajectory data
        df = pd.read_csv(date_file)
        logger.info(f"  Loaded {len(df)} trajectories")
        
        # Initialize time slot aggregator: {time_slot: {edge_id: [travel_times]}}
        time_slots = defaultdict(lambda: defaultdict(list))
        
        # Process each trajectory
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                logger.info(f"    Processed {idx}/{len(df)} trajectories...")
                
            try:
                path = ast.literal_eval(row['path'])
                timestamps = ast.literal_eval(row['timestamp'])
                pass_times = ast.literal_eval(row['pass_time'])
                
                # Process each segment in the path
                for i, edge_id in enumerate(path):
                    if i >= len(timestamps) or i >= len(pass_times):
                        continue
                        
                    timestamp = timestamps[i]
                    travel_time = pass_times[i]
                    
                    # Skip invalid data
                    if travel_time <= 0 or pd.isna(travel_time):
                        continue
                    
                    # Convert timestamp to time slot
                    dt = datetime.fromtimestamp(timestamp)
                    minutes = (dt.hour * 60 + dt.minute) // self.time_slot_minutes * self.time_slot_minutes
                    time_slot = f"{minutes//60:02d}:{minutes%60:02d}"
                    
                    # Add to aggregator
                    time_slots[time_slot][edge_id].append(travel_time)
                    
            except Exception as e:
                continue
        
        # Aggregate to DataFrame
        logger.info(f"  Aggregating {len(time_slots)} time slots...")
        
        # Generate all possible time slots for the day (00:00 to 23:55 in 5-min intervals)
        all_time_slots = []
        for h in range(24):
            for m in range(0, 60, self.time_slot_minutes):
                all_time_slots.append(f"{h:02d}:{m:02d}")
        
        # Build result DataFrame
        rows = []
        for time_slot in all_time_slots:
            row = {'time_slot': time_slot}
            
            # For each edge, compute mean travel time (or -1 if no data)
            for edge_id in range(self.num_edges):
                col_name = f'edge{edge_id}_traversal_time_sec'
                
                if edge_id in time_slots[time_slot] and len(time_slots[time_slot][edge_id]) > 0:
                    row[col_name] = np.mean(time_slots[time_slot][edge_id])
                else:
                    row[col_name] = -1.0  # Missing data sentinel
            
            rows.append(row)
        
        result_df = pd.DataFrame(rows)
        logger.info(f"  Created DataFrame with shape {result_df.shape}")
        
        return result_df
    
    def build_edge_connections(self):
        """
        Build edge_connections.csv from line_graph_edge_idx.npy
        Since we don't have actual vertex IDs, we'll create synthetic ones based on connectivity
        """
        logger.info("Building edge_connections.csv...")
        
        # Load line graph
        line_graph = np.load(self.didi_dir / 'line_graph_edge_idx.npy')
        logger.info(f"  Loaded line graph with {line_graph.shape[1]} connections")
        
        # Create synthetic vertex mapping
        # For each edge: if edge A connects to edge B in line graph, 
        # then vertex_end(A) = vertex_start(B)
        edge_start = {}
        edge_end = {}
        next_vertex_id = 0
        
        for i in range(line_graph.shape[1]):
            edge_from = line_graph[0, i]
            edge_to = line_graph[1, i]
            
            # If edge_from doesn't have an end vertex, create one
            if edge_from not in edge_end:
                edge_end[edge_from] = next_vertex_id
                next_vertex_id += 1
            
            # If edge_to doesn't have a start vertex, use edge_from's end vertex
            if edge_to not in edge_start:
                edge_start[edge_to] = edge_end[edge_from]
        
        # For edges without start/end vertices, create them
        for edge_id in range(self.num_edges):
            if edge_id not in edge_start:
                edge_start[edge_id] = next_vertex_id
                next_vertex_id += 1
            if edge_id not in edge_end:
                edge_end[edge_id] = next_vertex_id
                next_vertex_id += 1
        
        # Build DataFrame
        connections = []
        for edge_id in range(self.num_edges):
            connections.append({
                'edge_id': edge_id,
                'vertex_start_id': edge_start[edge_id],
                'vertex_end_id': edge_end[edge_id]
            })
        
        conn_df = pd.DataFrame(connections)
        output_path = self.output_dir / 'connections' / 'edge_connections.csv'
        conn_df.to_csv(output_path, index=False)
        logger.info(f"  Saved edge_connections.csv with {len(conn_df)} edges and {next_vertex_id} vertices")
        
        return conn_df
    
    def build_osm_metadata(self):
        """
        Convert edge_features.csv to osm_roads_output.json format
        """
        logger.info("Building osm_roads_output.json...")
        
        osm_data = {}
        
        for idx, row in self.edge_features.iterrows():
            edge_id = row['road_id']
            
            # Map to OSM format (as expected by StaticGraphBuilder)
            osm_data[str(edge_id)] = {
                'road_type': row['highway'],
                'oneway': bool(row['oneway']),
                'lanes': int(row['lanes']) if not pd.isna(row['lanes']) else 0,
                'length': float(row['length']),
                'bridge': bool(row['bridge']),
                'tunnel': bool(row['tunnel']),
                'nodes': [row['road_id'], row['road_id']]  # Dummy nodes (edge ID used)
            }
        
        output_path = self.output_dir / 'osm_data' / 'osm_roads_output.json'
        with open(output_path, 'w') as f:
            json.dump(osm_data, f, indent=2)
        
        logger.info(f"  Saved osm_roads_output.json with {len(osm_data)} road segments")
    
    def convert_all_days(self):
        """
        Convert all daily trajectory files to Harbin format
        """
        logger.info("Starting conversion of all daily files...")
        
        # Get all daily CSV files
        daily_files = sorted(self.didi_dir.glob('2016*.csv'))
        logger.info(f"Found {len(daily_files)} daily files to process")
        
        for i, date_file in enumerate(daily_files):
            logger.info(f"\n=== Processing day {i+1}/{len(daily_files)}: {date_file.name} ===")
            
            # Aggregate trajectories to time slots
            df = self.aggregate_trajectories_to_timeslots(date_file)
            
            # Save
            output_file = self.output_dir / 'days' / f'edge_data_day{i+1}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"  Saved to {output_file}")
        
        # Build connection and metadata files
        logger.info("\n=== Building auxiliary files ===")
        self.build_edge_connections()
        self.build_osm_metadata()
        
        logger.info("\n=== Conversion complete! ===")
        logger.info(f"Output directory: {self.output_dir}")


def main():
    didi_dir = 'src/TrainingData/ExternalData/didi_chengdu'
    output_dir = 'src/TrainingData/didi_chengdu_converted'
    
    converter = DidiToHarbinConverter(didi_dir, output_dir)
    converter.convert_all_days()


if __name__ == '__main__':
    main()
