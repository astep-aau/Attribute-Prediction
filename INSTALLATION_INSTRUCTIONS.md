# Installation Instructions for PyTorch Dependencies

## Issue
The evaluation script requires PyTorch and PyTorch Geometric, which are commented out in requirements.txt due to Windows long path limitations.

## Solution

### Step 1: Install PyTorch (CPU version)
Run from a **shorter path** to avoid Windows long path issues:

```powershell
cd C:\Users\jacob
& "C:/Users/jacob/OneDrive - Aalborg Universitet/AAU Uni/AAU 5. semester/P5/P5_Attribute_Prediction/Attribute-Prediction/.venv/Scripts/pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Wait for completion (this can take 5-10 minutes).

### Step 2: Install PyTorch Geometric
After PyTorch is installed:

```powershell
cd C:\Users\jacob
& "C:/Users/jacob/OneDrive - Aalborg Universitet/AAU Uni/AAU 5. semester/P5/P5_Attribute_Prediction/Attribute-Prediction/.venv/Scripts/pip.exe" install torch-geometric
```

### Step 3: Verify Installation
```powershell
cd "C:/Users/jacob/OneDrive - Aalborg Universitet/AAU Uni/AAU 5. semester/P5/P5_Attribute_Prediction/Attribute-Prediction"
& .venv/Scripts/python.exe -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__)"
```

### Alternative: Manual Installation
If pip installation fails due to path length issues:

1. Download PyTorch wheel from: https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp312-cp312-win_amd64.whl
2. Install locally:
   ```powershell
   pip install path/to/downloaded/torch-2.5.1+cpu-cp312-cp312-win_amd64.whl
   ```

## Changes Made to Evaluation Script

✅ **Added MAPE metric** to evaluation:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)  
- **MAPE (Mean Absolute Percentage Error)** - percentage-based metric
- Handles division by zero for MAPE calculation

Results now include:
```
model                                   mask_rate  MAE    RMSE   MAPE    num_masked
GAT_L1_LR0.0001_GNN200_GRU200_H1_D0.2  10%        X.XX   Y.YY   Z.ZZ%   NNNNN
```

## Running After Installation

Once PyTorch is installed, check conversion status and run evaluation:

```powershell
# Check if conversion is complete
Get-Content "src\TrainingData\didi_chengdu_converted\days\edge_data_day10.csv" -Head 5

# Run evaluation
& .venv/Scripts/python.exe evaluate_imputation.py
```
