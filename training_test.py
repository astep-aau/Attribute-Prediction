from src.data_manipulation.graph_dataset_builder import GraphDatasetBuilder
from src.data_manipulation.file_loader import FileLoader
from src.data_manipulation.temporal_dataset_builder import TemporalDatasetBuilder
from src.models.graphsage_gru import GraphSAGEGru
from pathlib import Path
import torch
import torch.nn as nn

print("=" * 60)
print("TESTING GRAPHSAGE-GRU MODEL")
print("=" * 60)

# ===== 1. Load Data =====
print("\n1. Loading data...")
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
print("   ✓ Data loaded")

# ===== 2. Build Graph =====
print("\n2. Building graph...")
graph_dataset = GraphDatasetBuilder(fileloader)
print(f"   ✓ Nodes: {graph_dataset.x.shape[0]}")
print(f"   ✓ Node features: {graph_dataset.x.shape[1]}")
print(f"   ✓ Edge connections: {graph_dataset.edge_index.shape[1]}")

# ===== 3. Build Temporal Dataset =====
print("\n3. Building temporal dataset...")
sequence_length = 5
temp_data_builder = TemporalDatasetBuilder(graph_dataset, sequence_length)
print(f"   ✓ Total sequences: {len(temp_data_builder)}")

# Get one sample
sample = temp_data_builder[0]
print(f"   ✓ Sample shape: {sample.shape}")
print(f"     [seq_len={sample.shape[0]}, num_nodes={sample.shape[1]}, features={sample.shape[2]}]")

# ===== 4. Initialize Model =====
print("\n4. Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

in_dim = sample.shape[2]  # Feature dimension from data
model = GraphSAGEGru(
    in_dim=in_dim,
    out_dim=1,  # Predicting travel time
    gnn_num_layers=2,
    gru_num_layers=1,
    gnn_hidden_dim=32,
    gru_hidden_dim=64,
    gnn_dropout=0.2,
    gru_dropout=0.2,
    gnn_agg_method='mean'
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"   ✓ Model created with {total_params:,} parameters")

# ===== 5. Test Forward Pass =====
print("\n5. Testing forward pass...")
edge_index = graph_dataset.edge_index.to(device)
sample = sample.to(device)

with torch.no_grad():
    output, hidden = model(sample, edge_index)
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Hidden shape: {hidden.shape}")
    print(f"   ✓ Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")

# ===== 6. Test Training Step with Real Targets =====
print("\n6. Testing training step with real targets...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Get real target from next timestep (sequence_length + prediction_horizon)
target_timestep = sequence_length  # Predicting timestep 5 from 0-4
target = graph_dataset._build_target_tensor(target_timestep).to(device)  # [num_nodes]

# Check how many valid measurements we have
valid_mask = target != -1.0
num_valid = valid_mask.sum().item()
num_total = len(target)
print(f"   Target timestep: {target_timestep}")
print(f"   Valid measurements: {num_valid}/{num_total} ({100*num_valid/num_total:.1f}%)")
print(f"   Target range (valid): [{target[valid_mask].min().item():.2f}, {target[valid_mask].max().item():.2f}] sec")

# Training step
model.train()
optimizer.zero_grad()
output, _ = model(sample, edge_index)

if num_valid > 0:
    # Only compute loss on valid measurements
    loss = criterion(output[valid_mask], target[valid_mask].unsqueeze(1))
    loss.backward()
    optimizer.step()

    # Calculate mean absolute error for interpretability
    with torch.no_grad():
        mae = torch.abs(output[valid_mask] - target[valid_mask].unsqueeze(1)).mean()

    print(f"   ✓ Loss (MSE): {loss.item():.4f}")
    print(f"   ✓ MAE: {mae.item():.2f} seconds")
else:
    print("   ⚠ No valid targets to compute loss")

# ===== 7. Test Multiple Training Steps =====
print("\n7. Testing mini training loop (3 steps)...")
model.train()

for step in range(3):
    # Get different sample and target
    idx = step * 10  # Use different timesteps
    if idx >= len(temp_data_builder):
        break

    sample = temp_data_builder[idx].to(device)
    target_timestep = idx + sequence_length
    target = graph_dataset._build_target_tensor(target_timestep).to(device)

    valid_mask = target != -1.0

    if valid_mask.sum() > 0:
        optimizer.zero_grad()
        output, _ = model(sample, edge_index)
        loss = criterion(output[valid_mask], target[valid_mask].unsqueeze(1))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mae = torch.abs(output[valid_mask] - target[valid_mask].unsqueeze(1)).mean()

        print(f"   Step {step+1}: Loss={loss.item():.4f}, MAE={mae.item():.2f}s, Valid={valid_mask.sum().item()}")

# ===== 8. Test Inference =====
print("\n8. Testing inference mode...")
model.eval()

with torch.no_grad():
    sample = temp_data_builder[0].to(device)
    target_timestep = sequence_length
    target = graph_dataset._build_target_tensor(target_timestep).to(device)

    output, _ = model(sample, edge_index)
    valid_mask = target != -1.0

    if valid_mask.sum() > 0:
        mae = torch.abs(output[valid_mask] - target[valid_mask].unsqueeze(1)).mean()

        # Show some example predictions
        num_examples = min(5, valid_mask.sum().item())
        valid_indices = torch.where(valid_mask)[0][:num_examples]

        print(f"   Sample predictions vs actual:")
        for i, idx in enumerate(valid_indices):
            pred = output[idx].item()
            actual = target[idx].item()
            error = abs(pred - actual)
            print(f"     Node {idx.item()}: Predicted={pred:.2f}s, Actual={actual:.2f}s, Error={error:.2f}s")

        print(f"   ✓ Overall MAE: {mae.item():.2f} seconds")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
