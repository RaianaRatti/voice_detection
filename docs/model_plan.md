# Custom VAD Model Plan
## Goal
Replace `webrtcvad` with a custom neural model that handles:
- Standard speech / silence detection (what WebRTC VAD already does)
- Overlapping voices (two people talking at once)
- Voices in close succession (back-to-back with no gap)
- Laughter and non-speech vocalizations (counted as speech, not silence)

---

## File Structure Changes

```
voice_detection/
├── audio/
│   ├── capture.py           (unchanged)
│   └── vad.py               (MODIFIED — swap WebRTC for custom model)
├── diarization/
│   ├── clustering.py        (unchanged)
│   ├── pipeline.py          (MODIFIED — minor, see Step 6)
│   └── speaker_tracker.py   (unchanged)
├── embeddings/
│   └── encoder.py           (unchanged)
├── features/
│   └── mfcc.py              (unchanged)
├── ml/                      (NEW FOLDER)
│   ├── dataset.py           (NEW — data loading + augmentation)
│   ├── model.py             (NEW — neural network definition)
│   ├── train.py             (NEW — training loop)
│   └── evaluate.py          (NEW — evaluation + threshold tuning)
├── data/
│   └── labels/              (NEW — your annotation CSVs go here)
├── models/
│   └── custom_vad.pt        (NEW — saved model weights after training)
├── server/
│   ├── app.py               (unchanged)
│   └── state.py             (unchanged)
└── config.py                (MODIFIED — add model path + threshold)
```

---

## Step 1 — Collect and Label Training Data

You need audio clips labeled at the frame level (every 30ms frame is one of 4 classes):

| Label | Int | Meaning |
|-------|-----|---------|
| silence | 0 | No voice activity |
| speech | 1 | One person speaking clearly |
| overlap | 2 | Two or more voices at once |
| vocalization | 3 | Laughter, coughing, non-speech sounds |

### Recommended datasets (free):

- **LibriSpeech** — clean single-speaker speech (labels: `speech`)
  - https://www.openslr.org/12
- **AMI Meeting Corpus** — overlapping speech in meetings (labels: `speech`, `overlap`)
  - https://groups.inf.ed.ac.uk/ami/corpus/
- **AudioSet** — laughter, coughing, crowd noise (labels: `vocalization`, `silence`)
  - https://research.google.com/audioset/
- **MUSAN** — noise and music (labels: `silence`)
  - https://www.openslr.org/17

### Label format

Create CSVs in `data/labels/` with this schema:

```
filename,start_ms,end_ms,label
meeting_001.wav,0,30,silence
meeting_001.wav,30,60,speech
meeting_001.wav,60,90,overlap
meeting_001.wav,90,120,vocalization
...
```

You can label your own recordings using **Audacity** (free):
- Open a WAV file
- Use Label Track (Tracks → Add New → Label Track)
- Export labels as text, then convert to the CSV format above

Aim for at least:
- 2 hours of `speech`
- 1 hour of `silence`
- 30 minutes of `overlap`
- 30 minutes of `vocalization`

---

## Step 2 — `ml/dataset.py` (NEW FILE)

Loads audio + labels, slices into 30ms frames, extracts features.

```python
# ml/dataset.py

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
N_MFCC = 40
N_MELS = 40
LABEL_MAP = {"silence": 0, "speech": 1, "overlap": 2, "vocalization": 3}

def extract_features(frame: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC + delta + delta-delta + energy from one 30ms frame."""
    if frame.dtype != np.float32:
        frame = frame.astype(np.float32) / 32768.0

    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=N_MFCC)          # (40, T)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    energy = np.array([[np.log(np.sum(frame ** 2) + 1e-8)]])

    features = np.concatenate([
        mfcc.mean(axis=1),
        delta.mean(axis=1),
        delta2.mean(axis=1),
        energy.flatten()
    ])  # shape: (121,)

    return features.astype(np.float32)


class VADDataset(Dataset):
    def __init__(self, label_csv: str, audio_dir: str, augment: bool = False):
        self.df = pd.read_csv(label_csv)
        self.audio_dir = Path(audio_dir)
        self.augment = augment
        self.samples = []
        self._load()

    def _load(self):
        for filename, group in self.df.groupby("filename"):
            audio_path = self.audio_dir / filename
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            audio = (audio * 32768).astype(np.int16)

            for _, row in group.iterrows():
                start = int(row["start_ms"] * SAMPLE_RATE / 1000)
                end = int(row["end_ms"] * SAMPLE_RATE / 1000)
                frame = audio[start:end]

                if len(frame) != FRAME_SIZE:
                    continue

                label = LABEL_MAP[row["label"]]
                self.samples.append((frame, label))

    def _augment(self, frame: np.ndarray) -> np.ndarray:
        """Simple augmentation: add gaussian noise or scale amplitude."""
        if np.random.rand() < 0.5:
            noise = np.random.randn(len(frame)).astype(np.float32) * 0.005 * 32768
            frame = np.clip(frame.astype(np.float32) + noise, -32768, 32767).astype(np.int16)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.7, 1.3)
            frame = np.clip(frame.astype(np.float32) * scale, -32768, 32767).astype(np.int16)
        return frame

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, label = self.samples[idx]
        if self.augment:
            frame = self._augment(frame)
        features = extract_features(frame)
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)
```

---

## Step 3 — `ml/model.py` (NEW FILE)

A small feedforward network — fast enough to run every 30ms in real time.

```python
# ml/model.py

import torch
import torch.nn as nn

INPUT_DIM = 121   # matches extract_features() output
HIDDEN_DIM = 256
NUM_CLASSES = 4   # silence, speech, overlap, vocalization

class VADNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(HIDDEN_DIM, 128),
            nn.ReLU(),

            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)  # returns raw logits, shape (batch, 4)
```

This is intentionally small — inference runs in under 1ms on CPU per frame, so it won't bottleneck your 30ms pipeline loop.

If you want to capture temporal context (what happened in the last few frames), you can swap the architecture for an LSTM later. Start with this and see if accuracy is sufficient.

---

## Step 4 — `ml/train.py` (NEW FILE)

```python
# ml/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from ml.dataset import VADDataset
from ml.model import VADNet

# --- config ---
LABEL_CSV   = "data/labels/all_labels.csv"
AUDIO_DIR   = "data/audio"
MODEL_OUT   = "models/custom_vad.pt"
EPOCHS      = 30
BATCH_SIZE  = 256
LR          = 1e-3
VAL_SPLIT   = 0.15
# --------------

def train():
    dataset = VADDataset(LABEL_CSV, AUDIO_DIR, augment=True)

    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = VADNet().to(device)

    # class weights to handle imbalanced data (overlap + vocalization are rarer)
    # adjust these based on your actual class counts
    weights = torch.tensor([0.5, 1.0, 2.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # --- train ---
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

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
```

Run training from the project root:
```bash
python -m ml.train
```

---

## Step 5 — `ml/evaluate.py` (NEW FILE)

After training, use this to check per-class accuracy and tune thresholds.

```python
# ml/evaluate.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from ml.dataset import VADDataset
from ml.model import VADNet

LABEL_CSV  = "data/labels/all_labels.csv"
AUDIO_DIR  = "data/audio"
MODEL_PATH = "models/custom_vad.pt"
LABEL_NAMES = ["silence", "speech", "overlap", "vocalization"]

def evaluate():
    dataset = VADDataset(LABEL_CSV, AUDIO_DIR, augment=False)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = VADNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits   = model(features)
            preds    = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()
```

Run from project root:
```bash
python -m ml.evaluate
```

Look at the confusion matrix — if `overlap` is being misclassified as `speech` a lot, you need more overlap training data.

---

## Step 6 — `audio/vad.py` (MODIFIED)

Replace WebRTC VAD with the custom model while keeping the same interface so `pipeline.py` barely changes.

```python
# audio/vad.py

import numpy as np
import torch
from pathlib import Path

from ml.model import VADNet
from ml.dataset import extract_features
from config import SAMPLE_RATE, FRAME_MS, MIN_SPEECH_MS, CUSTOM_VAD_MODEL_PATH

SILENCE_CLASS      = 0
SPEECH_CLASS       = 1
OVERLAP_CLASS      = 2
VOCALIZATION_CLASS = 3

# All of these count as "something is happening" (not silence)
ACTIVE_CLASSES = {SPEECH_CLASS, OVERLAP_CLASS, VOCALIZATION_CLASS}


class VAD:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = VADNet().to(self.device)
        self.model.load_state_dict(
            torch.load(CUSTOM_VAD_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

    def _predict(self, frame: np.ndarray) -> int:
        """Returns predicted class int for one 30ms frame."""
        features = extract_features(frame)
        tensor   = torch.tensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return logits.argmax(dim=1).item()

    def is_speech(self, frame_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        """Drop-in replacement for webrtcvad — returns True if frame is active."""
        frame = np.frombuffer(frame_bytes, dtype=np.int16)
        pred  = self._predict(frame)
        return pred in ACTIVE_CLASSES

    def get_class(self, frame_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> int:
        """Extended method — returns the raw class int if you want overlap/vocalization info."""
        frame = np.frombuffer(frame_bytes, dtype=np.int16)
        return self._predict(frame)
```

`is_speech()` has the exact same signature as before, so `pipeline.py` needs no changes at all for basic functionality. The `get_class()` method is there if you later want to do something special when overlap is detected (e.g. flag both speakers as active).

---

## Step 7 — `config.py` (MODIFIED)

Add after the existing constants:

```python
CUSTOM_VAD_MODEL_PATH = "models/custom_vad.pt"
```

---

## Step 8 — Fallback During Development

While you are still collecting data and training, keep `webrtcvad` available as a fallback. In `audio/vad.py` you can add a safety check:

```python
class VAD:
    def __init__(self):
        model_path = Path(CUSTOM_VAD_MODEL_PATH)
        if model_path.exists():
            # load custom model
            ...
        else:
            # fall back to webrtcvad
            import webrtcvad
            self._fallback = webrtcvad.Vad(VAD_AGGRESSIVENESS)
            self._use_fallback = True
```

This way the app still runs while you are training.

---

## Step 9 — Install New Dependencies

```bash
pip install scikit-learn pandas
```

Everything else (`torch`, `librosa`) you likely already have.

---

## Summary of Order of Operations

1. Download datasets listed in Step 1
2. Label your own recordings in Audacity if you want domain-specific data
3. Create `data/labels/all_labels.csv` combining all sources
4. Run `python -m ml.train` — expect 20-40 minutes depending on data size
5. Run `python -m ml.evaluate` — check confusion matrix, retrain if needed
6. Once `models/custom_vad.pt` exists, the new `vad.py` loads it automatically
7. Run `python app.py` as normal — everything else is unchanged