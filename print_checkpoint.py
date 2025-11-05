import torch
import sys
import pprint

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_checkpoint.py <path_to_checkpoint>")
        sys.exit(1)

    path = sys.argv[1]

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    print(f"\nLoaded checkpoint from: {path}\n")

    # Print only meaningful fields
    info = {
        "epoch": checkpoint.get("epoch"),
        "final_loss": checkpoint.get("loss"),
        "training_duration_formatted": checkpoint.get("training_duration_formatted"),
        "hyperparameters": checkpoint.get("hyperparameters"),
        "data_info": checkpoint.get("data_info"),
        "loss_history_length": len(checkpoint.get("loss_history", []))
    }

    pprint.pprint(info, sort_dicts=False)

if __name__ == "__main__":
    main()
