# Didi Chengdu Imputation Evaluation Pipeline

## Overview
This pipeline converts Didi Chengdu trajectory data to Harbin format and evaluates cross-city transfer learning performance of trained models with random missingness at 10%, 20%, and 30%.

## Pipeline Steps

### 1. Data Conversion (`convert_didi_to_harbin.py`)

**Input (Didi Chengdu format):**
- `20161010.csv` to `20161019.csv`: Daily trajectory files (~160k trajectories/day)
  - Each row: order_id, start_time, path (list of road_ids), timestamp, pass_time
- `edge_features.csv`: 6,639 road segments with OSM-style features
- `line_graph_edge_idx.npy`: Pre-computed line graph connectivity

**Output (Harbin format):**
- `days/edge_data_day1.csv` to `edge_data_day10.csv`: Time-series files
  - Columns: `time_slot`, `edge0_traversal_time_sec`, `edge1_traversal_time_sec`, ...
  - 288 rows per day (5-minute intervals from 00:00 to 23:55)
  - 6,640 columns (1 time_slot + 6,639 edges)
- `connections/edge_connections.csv`: Edge connectivity
  - Columns: edge_id, vertex_start_id, vertex_end_id
- `osm_data/osm_roads_output.json`: OSM metadata per edge
  - road_type, oneway, lanes, length, bridge, tunnel

**Process:**
1. Parse trajectories: extract (road_id, timestamp, travel_time) tuples
2. Aggregate to 5-minute time slots: compute mean travel time per edge per slot
3. Handle missing data: use -1.0 sentinel for edges with no observations
4. Build connectivity from line graph (synthetic vertex IDs based on edge connections)
5. Convert edge features to OSM JSON format

### 2. Imputation Evaluation (`evaluate_imputation.py`)

**Configuration:**
- Test dataset: Last day (day 10) from converted data
- Models: Trained Harbin checkpoints from `src/app/saved_models/`
- Mask rates: 10%, 20%, 30% random missingness
- Metrics: MAE and RMSE on masked values only

**Evaluation process per model:**
1. Load converted Chengdu data in Harbin format
2. Build static graph structure (line graph with OSM features)
3. Create PyTorch Geometric dataset (no pre-masking)
4. For each mask rate:
   - Apply random mask to non-missing values (-1.0 excluded)
   - Run model inference with masked input
   - Compute MAE/RMSE between predictions and ground truth at masked positions
5. Save results to `imputation_evaluation_results.csv`

## Cross-City Transfer Learning

**Why this works:**
- GNN/BiGRU architectures are graph-size agnostic (weights shared across nodes)
- Model learns spatial message-passing patterns, not graph-specific structure
- Chengdu (6,639 nodes) uses same features as Harbin (different node count OK)

**Expected insights:**
- Tests if Harbin-learned patterns generalize to different Chinese city
- Shows model robustness to domain shift (different traffic patterns, network topology)
- Compares performance across missingness levels (10%/20%/30%)

## Running the Pipeline

```bash
# 1. Convert Didi Chengdu to Harbin format (10-15 minutes)
python convert_didi_to_harbin.py

# 2. Run imputation evaluation (5-10 minutes per model)
python evaluate_imputation.py
```

## Output Files

- `src/TrainingData/didi_chengdu_converted/`: Converted dataset
- `imputation_evaluation_results.csv`: Evaluation metrics

Example results format:
```
model                                        mask_rate  MAE     RMSE    num_masked
GAT_L1_LR0.0001_GNN200_GRU200_H1_D0.2       10%        5.234   8.123   45678
GAT_L1_LR0.0001_GNN200_GRU200_H1_D0.2       20%        5.891   9.234   91356
GAT_L1_LR0.0001_GNN200_GRU200_H1_D0.2       30%        6.445   10.123  137034
```

## Technical Details

**Feature alignment:**
- Static features: road_type (one-hot), oneway (binary) → matches Harbin OSM format
- Dynamic features: travel_time_sec per time slot → matches Harbin
- Temporal: 5-minute intervals, day-of-week encoding → matches Harbin config

**Model compatibility:**
- Input: (batch, nodes, seq_len, features) where nodes=6,639 (Chengdu) vs ~20,000 (Harbin)
- GNN: Message passing operates on provided edge_index (graph structure)
- BiGRU: Processes each node's temporal sequence independently
- Output: Per-node predictions for next time step

**Masking strategy:**
- Random mask ensures unbiased evaluation (no spatial/temporal patterns)
- Excludes pre-existing missing values (-1.0) from masking
- Maintains original missingness distribution + adds artificial masks
