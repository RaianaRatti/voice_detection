import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from ml.dataset import VADDataset
from ml.model import VADNet

LABEL_CSV  = "train_data/labels/all_labels.csv"
AUDIO_DIR  = "train_data/audio"
MODEL_OUT  = "models/custom_vad.pt" # where to save trained model
EPOCHS     = 30
BATCH_SIZE = 256 # groups examples (256 given to model at same time)
LR         = 1e-3 # Learning rate = 0.001, controls how large weight updates are
VAL_SPLIT  = 0.15 # 15% of data is validation data


def train():

    print("Creating dataset...")
    dataset = VADDataset("preprocessed_features/features.npy", 
                     "preprocessed_features/labels.npy", 
                     augment=True)
    print("Dataset created")
    print("Dataset size:", len(dataset))

    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    print("Creating dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0) # 4
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # 4
    print("Dataloaders created")

    print("Testing first batch...")
    for features, labels in train_loader:
        print("First batch loaded!")
        print("Features shape:", features.shape)
        print("Labels shape:", labels.shape)
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    print("Creating model...")
    model  = VADNet().to(device) # creates nn (moves to CPU or GPU)
    print("Model created")

    # vocalization and overlap are rarer so upweighting them
    weights = torch.tensor([1.5, 1.5, 2.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")

    print("Starting training loop...")
    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch+1}")
        model.train()
        print(f"Ending epoch {epoch+1}")
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Starting epoch {epoch+1} evaluation")
        model.eval()
        print(f"Ending epoch {epoch+1} evaluation")
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits    = model(features)
                val_loss += criterion(logits, labels).item()
                preds     = logits.argmax(dim=1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        acc       = correct / total * 100
        print(f"Epoch {epoch+1:02d} | train={avg_train:.4f} | val={avg_val:.4f} | acc={acc:.1f}%")
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            Path(MODEL_OUT).parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ✓ saved best model → {MODEL_OUT}")

    print("Training complete.")


if __name__ == "__main__":
    train()

'''
Notes:
    1. DataLoader --> handles batching, loading, shuffling
        i. We keep shuffle = True so model does not learn falsely on order
        ii. We batch 256 examples at a time (faster)
        iii. We use 4 workers at a time (faster)
    
    2. Prefetching --> Before training, PyTorch must read audio, extract_features, convert to tensors, build batches. CPU would be sitting idle while this is being done if num_workers = 0 

    3. Batch Size --> larger batch size with same amount of samples means less chance for model to adjust its weights
'''