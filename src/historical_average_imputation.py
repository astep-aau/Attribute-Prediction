"""
Historical-average imputation baseline for Harbin and Chengdu datasets.
- Uses a 10% random mask on valid observations of the target day.
- Predicts masked values with the historical mean (same edge, same time_slot) from other days.
- Reports MAE, RMSE, and MAPE.
"""
from pathlib import Path
from typing import List
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAR_DAY_FOLDER = PROJECT_ROOT / "src" / "TrainingData" / "days"
CHENGDU_DAY_FOLDER = PROJECT_ROOT / "src" / "TrainingData" / "didi_chengdu_converted" / "days"
RESULTS_PATH = PROJECT_ROOT / "imputation_results_historical_average.csv"
MASK_RATE = 0.10
RANDOM_SEED = 42


def get_master_columns(day_files: List[Path]) -> List[str]:
    """Collect the full set of edge columns across provided day files."""
    edge_cols = set()
    for path in day_files:
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue
        cols = [c for c in df.columns if c.startswith("edge") and c.endswith("sec")]
        edge_cols.update(cols)

    ordered_edges = sorted(edge_cols, key=lambda x: int(x.split("_")[0].replace("edge", "")))
    return ["time_slot"] + ordered_edges


def load_day(path: Path, master_cols: List[str]) -> pd.DataFrame:
    """Load a day file and pad missing edge columns with NaN, preserving column order."""
    df = pd.read_csv(path)
    missing = [c for c in master_cols if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan
    return df[master_cols]


def sample_mask(values: np.ndarray, mask_rate: float, seed: int) -> np.ndarray:
    """Create a boolean mask over valid entries using the requested masking rate."""
    rng = np.random.default_rng(seed)
    valid = (~np.isnan(values)) & (values != -1)
    valid_idx = np.flatnonzero(valid)
    num_to_mask = int(len(valid_idx) * mask_rate)

    mask_flat = np.zeros(values.size, dtype=bool)
    if num_to_mask > 0:
        chosen = rng.choice(valid_idx, size=num_to_mask, replace=False)
        mask_flat[chosen] = True

    return mask_flat.reshape(values.shape)


def compute_historical_average(histories: List[np.ndarray]) -> np.ndarray:
    """Compute historical mean per time_slot/edge, ignoring -1 and NaN."""
    stack = np.stack(histories)  # (D, T, E)
    stack = np.where((stack == -1) | np.isnan(stack), np.nan, stack)

    ha = np.nanmean(stack, axis=0)  # (T, E)

    # Fill gaps with per-edge mean, then global mean if needed.
    per_edge_mean = np.nanmean(stack.reshape(-1, stack.shape[2]), axis=0)
    ha = np.where(np.isnan(ha), per_edge_mean, ha)
    global_mean = float(np.nanmean(stack)) if not np.isnan(np.nanmean(stack)) else 0.0
    ha = np.where(np.isnan(ha), global_mean, ha)

    return ha


def compute_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> dict:
    """Return MAE, RMSE, and MAPE for aligned arrays."""
    if true_vals.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "count": 0}

    diff = true_vals - pred_vals
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    non_zero = true_vals != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((true_vals[non_zero] - pred_vals[non_zero]) / true_vals[non_zero])) * 100)
    else:
        mape = np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "count": int(true_vals.size)}


def run_dataset(name: str, day_folder: Path, target_day: int, mask_rate: float) -> dict:
    day_files = sorted(day_folder.glob("edge_data_day*.csv"))
    if not day_files:
        raise FileNotFoundError(f"No day files found in {day_folder}")

    target_path = day_folder / f"edge_data_day{target_day}.csv"
    if not target_path.exists():
        raise FileNotFoundError(f"Target day file missing: {target_path}")

    master_cols = get_master_columns(day_files)
    edge_cols = master_cols[1:]

    target_df = load_day(target_path, master_cols).set_index("time_slot")
    target_df = target_df.reindex(sorted(target_df.index))
    target_values = target_df[edge_cols].to_numpy(dtype=float)

    histories = []
    for path in day_files:
        if path == target_path:
            continue
        hist_df = load_day(path, master_cols).set_index("time_slot")
        hist_df = hist_df.reindex(target_df.index)
        histories.append(hist_df[edge_cols].to_numpy(dtype=float))

    if not histories:
        raise ValueError("Historical data is empty; need at least one other day.")

    ha = compute_historical_average(histories)

    mask = sample_mask(target_values, mask_rate, RANDOM_SEED)
    true_masked = target_values[mask]
    pred_masked = ha[mask]

    valid = (~np.isnan(true_masked)) & (~np.isnan(pred_masked)) & (true_masked != -1)
    metrics = compute_metrics(true_masked[valid], pred_masked[valid])

    logger.info(
        "%s day %s -> masked=%d MAE=%.4f RMSE=%.4f MAPE=%.2f%%",
        name,
        target_day,
        metrics["count"],
        metrics["MAE"],
        metrics["RMSE"],
        metrics["MAPE"],
    )

    return {
        "dataset": name,
        "target_day": target_day,
        "mask_rate": mask_rate,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "num_masked_points": metrics["count"],
    }


def main():
    configs = [
        {"name": "Harbin", "folder": HAR_DAY_FOLDER, "target_day": 5},
        {"name": "Chengdu", "folder": CHENGDU_DAY_FOLDER, "target_day": 9},
    ]

    results = []
    for cfg in configs:
        result = run_dataset(cfg["name"], cfg["folder"], cfg["target_day"], MASK_RATE)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    logger.info("Results saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
