"""
Batch training script to train multiple models sequentially.
Useful for running experiments with different configurations.
"""

from src.training.train_generic import train_model
from configs.model_configs import MODEL_CONFIGS
import torch
import time
from pathlib import Path


def train_all_models(config_names=None, device='auto'):
    """
    Train multiple models sequentially.

    Args:
        config_names: List of config names to train. If None, trains all.
        device: Device to use for training
    """
    if config_names is None:
        config_names = list(MODEL_CONFIGS.keys())

    results = {}
    total_start = time.time()

    print("=" * 80)
    print("BATCH TRAINING MULTIPLE MODELS")
    print("=" * 80)
    print(f"\nModels to train: {len(config_names)}")
    for name in config_names:
        print(f"  - {name}: {MODEL_CONFIGS[name]['model_name']}")
    print()

    for i, config_name in enumerate(config_names, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(config_names)}: {config_name}")
        print(f"{'='*80}\n")

        config = MODEL_CONFIGS[config_name]

        try:
            model, history, checkpoint = train_model(config, device=device)

            results[config_name] = {
                'success': True,
                'val_mae': checkpoint['val_mae'],
                'val_loss': checkpoint['val_loss'],
                'train_loss': checkpoint['train_loss'],
                'best_epoch': checkpoint['epoch'] + 1,
                'gap_pct': 100 * (checkpoint['val_loss'] - checkpoint['train_loss']) / checkpoint['train_loss']
            }

            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                del model
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n ERROR training {config_name}: {str(e)}")
            results[config_name] = {
                'success': False,
                'error': str(e)
            }

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nTotal time: {total_time/60:.1f} minutes\n")

    print(f"{'Model':<25} {'Status':<10} {'Val MAE (s)':<12} {'Val Loss':<12} {'Gap %':<10}")
    print("-" * 80)

    for config_name, result in results.items():
        model_name = MODEL_CONFIGS[config_name]['model_name']
        if result['success']:
            print(f"{model_name:<25} {'Success':<10} {result['val_mae']:<12.2f} "
                  f"{result['val_loss']:<12.4f} {result['gap_pct']:<10.1f}")
        else:
            print(f"{model_name:<25} {'Failed':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print("\n" + "=" * 80)

    # Find best model
    successful_models = {k: v for k, v in results.items() if v['success']}
    if successful_models:
        best_model = min(successful_models.items(), key=lambda x: x[1]['val_mae'])
        print(f"\n Best Model: {MODEL_CONFIGS[best_model[0]]['model_name']}")
        print(f"   Validation MAE: {best_model[1]['val_mae']:.2f} seconds")
        print(f"   Model file: best_model_{best_model[0]}.pth")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train multiple models in batch')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='Specific configs to train (default: all)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to train on')

    args = parser.parse_args()

    if args.configs:
        # Validate config names
        invalid = [c for c in args.configs if c not in MODEL_CONFIGS]
        if invalid:
            print(f"Error: Invalid config names: {invalid}")
            print(f"Valid configs: {list(MODEL_CONFIGS.keys())}")
            exit(1)

    train_all_models(config_names=args.configs, device=args.device)
