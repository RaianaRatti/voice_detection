import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from ml.dataset import VADDataset
from ml.model import VADNet

LABEL_CSV   = "train_data/labels/all_labels.csv"
AUDIO_DIR   = "train_data/audio"
MODEL_PATH  = "models/custom_vad.pt"
LABEL_NAMES = ["silence", "speech", "overlap", "vocalization"]


def evaluate():
    dataset = VADDataset(LABEL_CSV, AUDIO_DIR, augment=False)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VADNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()