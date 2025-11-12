import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import time

from lstm_model import LSTMModel
from data_manipulation import DataManipulation

# Data
# arr = [25, 50, 75, 100, 125, 150, 200]

# for value in arr:
#    dm = DataManipulation('data/edge_traversals_processed.json', value, 50)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Sequence Length is about the structure of one sample.
Sequence_Length = 50
dm = DataManipulation('data/edge_traversals_processed.json', 125, Sequence_Length)

model = LSTMModel(input_dim=1, hidden_dim=75, output_dim=1, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training

# Batch Size is about how many samples you process in parallel.
# determined by your GPU's memory (bigger is often faster, but uses more memory).
batch_size = 64

train_dataset = TensorDataset(dm.get_trainX(), dm.get_trainY())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 100
epoch_losses = [] # A list to store the average loss of each epoch

start_time = time.time()
print("Starting training...")
for epoch in range(num_epochs):
    model.train() # Set the model to training mode

    running_loss = 0.0

    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        # Standard training steps hello
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(avg_epoch_loss)
    epoch_time = time.time() - start_time
    epoch_mins = int(epoch_time // 60)
    epoch_secs = int(epoch_time % 60)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.6f}, Time: {epoch_mins} minutes, {epoch_secs} seconds")


print("Training finished.")

end_time = time.time()

training_duration_seconds = end_time - start_time

mins = int(training_duration_seconds // 60)
secs = int(training_duration_seconds % 60)
print(f"\nTraining finished.")
print(f"Total training time: {mins} minutes, {secs} seconds")

# --- SAVING THE MODEL AND METADATA ---
pathtime = time.time()
filepath = str(pathtime) + "lstm_model_checkpoint.pth"

final_loss = epoch_losses[-1]

checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': final_loss,
    'training_duration_seconds': training_duration_seconds,
    'training_duration_formatted': f"{mins}m {secs}s",
    'hyperparameters': {
        'input_dim': model.lstm.input_size,
        'hidden_dim': model.hidden_size,
        'num_layers': model.num_layers,
        'output_dim': model.fc.out_features,
        'sequence_length': Sequence_Length,
        'batch_size': batch_size,
        'learning_rate': optimizer.param_groups[0]['lr']
    },
    'data_info': {
        'min_event_threshold': dm._min_event_threshold,
        'truncation_length': len(dm.df_data),
        'number_of_roads': len(dm.df_data.columns)
    },
    'loss_history': epoch_losses
}

# Save the dictionary
torch.save(checkpoint, filepath)

print(f"\nModel and metadata saved to {filepath}")
