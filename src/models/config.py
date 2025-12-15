import torch

TRAINING_DATA_FOLDER_NAME = "TrainingData"

# Data Params
SEQ_LEN = 12
BATCH_SIZE = 1
MASK_RATE = 0.2

# Model Architecture Params
GAT_HIDDEN_DIM = 128
GRU_HIDDEN_DIM = 128
GRU_LAYERS = 2
GAT_HEADS = 2
DROPOUT = 0.3

# Training Params
NUM_EPOCHS = 1
LEARNING_RATE = 0.002
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')