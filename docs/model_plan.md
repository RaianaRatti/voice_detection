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
│   ├── capture.py               (unchanged)
│   └── vad.py                   (MODIFIED — swap WebRTC for custom model)
├── diarization/
│   ├── clustering.py            (unchanged)
│   ├── pipeline.py              (MODIFIED — minor, see Step 6)
│   └── speaker_tracker.py       (unchanged)
├── embeddings/
│   └── encoder.py               (unchanged)
├── features/
│   └── mfcc.py                  (unchanged)
├── ml/                          (NEW FOLDER)
│   ├── labeling/                (NEW FOLDER)
│   │   ├── label_librispeech.py (NEW — auto-labels LibriSpeech frames)
│   │   ├── label_ami.py         (NEW — auto-labels AMI frames)
│   │   └── label_esc50.py       (NEW — auto-labels ESC-50 frames)
│   ├── merge_labels.py          (NEW — combines all CSVs into all_labels.csv)
│   ├── dataset.py               (NEW — data loading + augmentation)
│   ├── model.py                 (NEW — neural network definition)
│   ├── train.py                 (NEW — training loop)
│   └── evaluate.py              (NEW — evaluation + threshold tuning)
├── train_data/                  (NEW FOLDER — all training data lives here)
│   ├── audio/
│   │   ├── librispeech/         (extracted train-clean-100.tar.gz)
│   │   ├── librispeech_flat/    (auto-created by label_librispeech.py)
│   │   ├── ami/
│   │   │   ├── amicorpus/       (headset WAVs downloaded by wget script)
│   │   │   └── ami_public_manual_1.6.2/  (unzipped manual annotations)
│   │   └── ESC-50-master/       (unzipped from GitHub)
│   └── labels/
│       ├── librispeech_labels.csv
│       ├── ami_labels.csv
│       ├── esc50_labels.csv
│       └── all_labels.csv       (merged, created by merge_labels.py)
├── models/
│   └── custom_vad.pt            (NEW — saved model weights after training)
├── server/
│   ├── app.py                   (unchanged)
│   └── state.py                 (unchanged)
└── config.py                    (MODIFIED — add model path)
```

---

## Step 1 — Collect Training Data

You need audio clips labeled at the frame level (every 30ms frame is one of 4 classes):

| Label | Int | Meaning |
|-------|-----|---------|
| silence | 0 | No voice activity |
| speech | 1 | One person speaking clearly |
| overlap | 2 | Two or more voices at once |
| vocalization | 3 | Laughter, coughing, non-speech sounds |

### Datasets used:

- **LibriSpeech** (`train-clean-100.tar.gz`, 6.3GB) — labels: `speech`, `silence`
  - Download: https://www.openslr.org/12
  - Extract into: `train_data/audio/librispeech/`
  - Only process first ~200 files (~2 hours) via cap in labeling script

- **AMI Meeting Corpus** — labels: `speech`, `overlap`, `vocalization`
  - Audio: run the wget shell script from https://groups.inf.ed.ac.uk/ami/download/
    - Select meetings: ES2008, ES2009, ES2010 (all a/b/c/d), media: Individual headsets
    - Extract into: `train_data/audio/ami/amicorpus/`
  - Annotations: download `AMI manual annotations v1.6.2` zip from same page
    - Unzip into: `train_data/audio/ami/ami_public_manual_1.6.2/`
  - Vocalsound tags are embedded in the `words/` XML files (no separate folder)

- **ESC-50** — labels: `vocalization` (laughing, coughing)
  - Download: `wget https://github.com/karoldvl/ESC-50/archive/master.zip`
  - Extract into: `train_data/audio/ESC-50-master/`
  - 40 clips × 5 seconds per category = ~6,600 vocalization frames

### Label format (all CSVs share this schema):
```
filename,start_ms,end_ms,label
meeting_001.wav,0,30,silence
meeting_001.wav,30,60,speech
meeting_001.wav,60,90,overlap
meeting_001.wav,90,120,vocalization
```

---

## Step 2 — Auto-Label Each Dataset

### `ml/labeling/label_librispeech.py` (NEW FILE)

Uses webrtcvad (reliable on clean audio) to auto-label each 30ms frame as
`speech` or `silence`. Balances classes and caps at ~200 files.

```python
# ml/labeling/label_librispeech.py
# Run from project root: python -m ml.labeling.label_librispeech

import librosa
import numpy as np
import pandas as pd
import webrtcvad
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

SAMPLE_RATE   = 16000
FRAME_MS      = 30
FRAME_SIZE    = int(SAMPLE_RATE * FRAME_MS / 1000)

LIBRISPEECH_DIR = "train_data/audio/librispeech"
OUTPUT_CSV      = "train_data/labels/librispeech_labels.csv"

vad = webrtcvad.Vad(1)


def label_file(wav_path: Path) -> list[dict]:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    audio_int16 = (audio * 32768).astype(np.int16)
    rows = []

    for i in range(0, len(audio_int16) - FRAME_SIZE, FRAME_SIZE):
        frame = audio_int16[i:i + FRAME_SIZE]
        if len(frame) != FRAME_SIZE:
            continue
        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        rows.append({
            "filename": wav_path.name,
            "start_ms": int(i / SAMPLE_RATE * 1000),
            "end_ms":   int((i + FRAME_SIZE) / SAMPLE_RATE * 1000),
            "label":    "speech" if is_speech else "silence"
        })
    return rows


def run():
    wav_files = list(Path(LIBRISPEECH_DIR).rglob("*.flac"))
    print(f"Found {len(wav_files)} audio files")
    wav_files = wav_files[:200]  # ~2 hours; increase later if needed
    print(f"Using {len(wav_files)} files")

    flat_dir = Path("train_data/audio/librispeech_flat")
    flat_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for wav_path in tqdm(wav_files):
        flat_path = flat_dir / (wav_path.stem + ".wav")
        if not flat_path.exists():
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            sf.write(flat_path, audio, SAMPLE_RATE)
        all_rows.extend(label_file(flat_path))

    df = pd.DataFrame(all_rows)
    speech_count  = (df["label"] == "speech").sum()
    silence_count = (df["label"] == "silence").sum()
    print(f"Before balancing — speech: {speech_count}, silence: {silence_count}")

    silence_df  = df[df["label"] == "silence"].sample(n=min(silence_count, speech_count), random_state=42)
    speech_df   = df[df["label"] == "speech"]
    df_balanced = pd.concat([speech_df, silence_df]).sample(frac=1, random_state=42)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_balanced)} labeled frames → {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
```

Run:
```bash
python -m ml.labeling.label_librispeech
```

---

### `ml/labeling/label_ami.py` (NEW FILE)

Parses the AMI words XML files to get per-speaker timestamps. Detects overlap
when 2+ speakers are active in the same 30ms frame. Detects vocalization from
`<vocalsound>` tags.

```python
# ml/labeling/label_ami.py
# Run from project root: python -m ml.labeling.label_ami

import xml.etree.ElementTree as ET
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SAMPLE_RATE  = 16000
FRAME_MS     = 30
FRAME_SIZE   = int(SAMPLE_RATE * FRAME_MS / 1000)

AMI_AUDIO_DIR = "train_data/audio/ami/amicorpus"
AMI_WORDS_DIR = "train_data/audio/ami/ami_public_manual_1.6.2/words"
OUTPUT_CSV    = "train_data/labels/ami_labels.csv"

MEETINGS = [
    "ES2008a", "ES2008b", "ES2008c", "ES2008d",
    "ES2009a", "ES2009b", "ES2009c", "ES2009d",
    "ES2010a", "ES2010b", "ES2010c", "ES2010d",
]
SPEAKERS = ["A", "B", "C", "D"]


def parse_words_xml(xml_path: Path) -> tuple[list, list]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    speech_segments = []
    vocal_segments  = []

    for elem in root:
        tag   = elem.tag.split("}")[-1]
        start = elem.get("starttime")
        end   = elem.get("endtime")
        if start is None or end is None:
            continue
        start, end = float(start), float(end)
        if end <= start:
            continue
        if tag == "w":
            speech_segments.append((start, end))
        elif tag == "vocalsound":
            vocal_segments.append((start, end))

    return speech_segments, vocal_segments


def label_meeting(meeting_id: str) -> list[dict]:
    all_speech, all_vocal = [], []

    for speaker in SPEAKERS:
        xml_path = Path(AMI_WORDS_DIR) / f"{meeting_id}.{speaker}.words.xml"
        if not xml_path.exists():
            continue
        s, v = parse_words_xml(xml_path)
        all_speech.extend(s)
        all_vocal.extend(v)

    if not all_speech and not all_vocal:
        print(f"  No annotations found for {meeting_id}, skipping.")
        return []

    audio_path = None
    for h in range(4):
        candidate = Path(AMI_AUDIO_DIR) / meeting_id / "audio" / f"{meeting_id}.Headset-{h}.wav"
        if candidate.exists():
            audio_path = candidate
            break

    if audio_path is None:
        print(f"  No audio found for {meeting_id}, skipping.")
        return []

    duration_sec  = librosa.get_duration(path=str(audio_path))
    total_frames  = int(duration_sec * 1000 / FRAME_MS)
    speech_count  = np.zeros(total_frames, dtype=np.int16)
    vocal_count   = np.zeros(total_frames, dtype=np.int16)

    def mark(segments, counter):
        for start, end in segments:
            f_start = int(start * 1000 / FRAME_MS)
            f_end   = min(int(end * 1000 / FRAME_MS) + 1, total_frames)
            counter[f_start:f_end] += 1

    mark(all_speech, speech_count)
    mark(all_vocal,  vocal_count)

    rows = []
    filename = f"{meeting_id}.Headset-0.wav"

    for i in range(total_frames):
        sc = speech_count[i]
        vc = vocal_count[i]
        if sc >= 2:
            label = "overlap"
        elif sc == 1:
            label = "speech"
        elif vc >= 1:
            label = "vocalization"
        else:
            label = "silence"

        rows.append({
            "filename": filename,
            "start_ms": i * FRAME_MS,
            "end_ms":   i * FRAME_MS + FRAME_MS,
            "label":    label
        })

    return rows


def run():
    all_rows = []

    for meeting_id in tqdm(MEETINGS, desc="Processing meetings"):
        rows = label_meeting(meeting_id)
        all_rows.extend(rows)
        if rows:
            counts = pd.DataFrame(rows)["label"].value_counts().to_dict()
            print(f"  {meeting_id}: {counts}")

    if not all_rows:
        print("No rows generated — check your paths.")
        return

    df = pd.DataFrame(all_rows)
    print("\nRaw label counts:")
    print(df["label"].value_counts())

    overlap_count = (df["label"] == "overlap").sum()
    cap = max(overlap_count * 3, 10000)

    balanced_parts = []
    for label in ["silence", "speech", "overlap", "vocalization"]:
        subset = df[df["label"] == label]
        if len(subset) > cap and label in ("silence", "speech"):
            subset = subset.sample(n=cap, random_state=42)
        balanced_parts.append(subset)

    df_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"\nBalanced label counts:")
    print(df_balanced["label"].value_counts())
    print(f"Saved {len(df_balanced)} labeled frames → {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
```

Run:
```bash
python -m ml.labeling.label_ami
```

---

### `ml/labeling/label_esc50.py` (NEW FILE)

Slices each ESC-50 clip into 30ms frames and labels them all as `vocalization`.
Only processes `laughing` and `coughing` categories.

```python
# ml/labeling/label_esc50.py
# Run from project root: python -m ml.labeling.label_esc50

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

SAMPLE_RATE = 16000
FRAME_MS    = 30
FRAME_SIZE  = int(SAMPLE_RATE * FRAME_MS / 1000)

ESC50_AUDIO_DIR = "train_data/audio/ESC-50-master/audio"
ESC50_CSV       = "train_data/audio/ESC-50-master/meta/esc50.csv"
OUTPUT_CSV      = "train_data/labels/esc50_labels.csv"

CATEGORY_MAP = {
    "laughing": "vocalization",
    "coughing": "vocalization",
}


def extract_frames(wav_path: Path) -> list[np.ndarray]:
    audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    audio_int16 = (audio * 32768).astype(np.int16)
    frames = []
    for i in range(0, len(audio_int16) - FRAME_SIZE, FRAME_SIZE):
        frame = audio_int16[i:i + FRAME_SIZE]
        if len(frame) == FRAME_SIZE:
            frames.append(frame)
    return frames


def run():
    df_meta = pd.read_csv(ESC50_CSV)
    df_meta = df_meta[df_meta["category"].isin(CATEGORY_MAP.keys())]
    print(f"Found {len(df_meta)} clips: {df_meta['category'].value_counts().to_dict()}")

    rows = []
    audio_dir = Path(ESC50_AUDIO_DIR)

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        wav_path = audio_dir / row["filename"]
        if not wav_path.exists():
            print(f"  Missing: {wav_path}")
            continue
        label  = CATEGORY_MAP[row["category"]]
        frames = extract_frames(wav_path)
        for i, _ in enumerate(frames):
            rows.append({
                "filename": row["filename"],
                "start_ms": i * FRAME_MS,
                "end_ms":   i * FRAME_MS + FRAME_MS,
                "label":    label
            })

    df_out = pd.DataFrame(rows)
    print(f"\nLabel counts:\n{df_out['label'].value_counts()}")

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_out)} labeled frames → {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
```

Run:
```bash
python -m ml.labeling.label_esc50
```

---

## Step 3 — Merge All Labels

### `ml/merge_labels.py` (NEW FILE)

Combines all three CSVs into one `all_labels.csv` ready for training.

```python
# ml/merge_labels.py
# Run from project root: python -m ml.merge_labels

import pandas as pd
from pathlib import Path

INPUT_CSVS = [
    "train_data/labels/librispeech_labels.csv",
    "train_data/labels/ami_labels.csv",
    "train_data/labels/esc50_labels.csv",
]
OUTPUT_CSV = "train_data/labels/all_labels.csv"


def run():
    dfs = []
    for csv_path in INPUT_CSVS:
        p = Path(csv_path)
        if not p.exists():
            print(f"WARNING: missing {csv_path} — skipping")
            continue
        df = pd.read_csv(p)
        print(f"{p.name}: {len(df)} rows — {df['label'].value_counts().to_dict()}")
        dfs.append(df)

    combined = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nFinal combined label counts:")
    print(combined["label"].value_counts())
    print(f"Total: {len(combined)} frames")

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
```

Run:
```bash
python -m ml.merge_labels
```

---

## Step 4 — `ml/dataset.py` (NEW FILE)

Loads audio + labels, slices into 30ms frames, extracts MFCC features.

```python
# ml/dataset.py

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_MS    = 30
FRAME_SIZE  = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
N_MFCC      = 40
LABEL_MAP   = {"silence": 0, "speech": 1, "overlap": 2, "vocalization": 3}


def extract_features(frame: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC + delta + delta-delta + energy from one 30ms frame."""
    if frame.dtype != np.float32:
        frame = frame.astype(np.float32) / 32768.0

    mfcc    = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=N_MFCC)
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)
    energy  = np.array([[np.log(np.sum(frame ** 2) + 1e-8)]])

    features = np.concatenate([
        mfcc.mean(axis=1),
        delta.mean(axis=1),
        delta2.mean(axis=1),
        energy.flatten()
    ])  # shape: (121,)

    return features.astype(np.float32)


class VADDataset(Dataset):
    def __init__(self, label_csv: str, audio_dir: str, augment: bool = False):
        self.df        = pd.read_csv(label_csv)
        self.audio_dir = Path(audio_dir)
        self.augment   = augment
        self.samples   = []
        self._load()

    def _load(self):
        for filename, group in self.df.groupby("filename"):
            audio_path = self.audio_dir / filename
            audio, _   = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            audio      = (audio * 32768).astype(np.int16)

            for _, row in group.iterrows():
                start = int(row["start_ms"] * SAMPLE_RATE / 1000)
                end   = int(row["end_ms"]   * SAMPLE_RATE / 1000)
                frame = audio[start:end]
                if len(frame) != FRAME_SIZE:
                    continue
                label = LABEL_MAP[row["label"]]
                self.samples.append((frame, label))

    def _augment(self, frame: np.ndarray) -> np.ndarray:
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

## Step 5 — `ml/model.py` (NEW FILE)

Small feedforward network — inference under 1ms per frame on CPU.

```python
# ml/model.py

import torch
import torch.nn as nn

INPUT_DIM   = 121  # matches extract_features() output
HIDDEN_DIM  = 256
NUM_CLASSES = 4    # silence, speech, overlap, vocalization


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
        return self.net(x)  # raw logits, shape (batch, 4)
```

---

## Step 6 — `ml/train.py` (NEW FILE)

```python
# ml/train.py
# Run from project root: python -m ml.train

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from ml.dataset import VADDataset
from ml.model import VADNet

LABEL_CSV  = "train_data/labels/all_labels.csv"
AUDIO_DIR  = "train_data/audio"
MODEL_OUT  = "models/custom_vad.pt"
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 1e-3
VAL_SPLIT  = 0.15


def train():
    dataset    = VADDataset(LABEL_CSV, AUDIO_DIR, augment=True)
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model  = VADNet().to(device)

    # vocalization and overlap are rarer — upweight them
    weights   = torch.tensor([0.5, 1.0, 2.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
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
```

Run:
```bash
python -m ml.train
```

---

## Step 7 — `ml/evaluate.py` (NEW FILE)

```python
# ml/evaluate.py
# Run from project root: python -m ml.evaluate

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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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
```

Run:
```bash
python -m ml.evaluate
```

If `overlap` is being misclassified as `speech` heavily, collect more AMI data.

---

## Step 8 — `audio/vad.py` (MODIFIED)

Drop-in replacement — same `is_speech()` signature so `pipeline.py` is unchanged.

```python
# audio/vad.py

import numpy as np
import torch
from pathlib import Path

from ml.model import VADNet
from ml.dataset import extract_features
from config import SAMPLE_RATE, CUSTOM_VAD_MODEL_PATH, VAD_AGGRESSIVENESS

SILENCE_CLASS      = 0
SPEECH_CLASS       = 1
OVERLAP_CLASS      = 2
VOCALIZATION_CLASS = 3
ACTIVE_CLASSES     = {SPEECH_CLASS, OVERLAP_CLASS, VOCALIZATION_CLASS}


class VAD:
    def __init__(self):
        model_path = Path(CUSTOM_VAD_MODEL_PATH)
        if model_path.exists():
            self._use_fallback = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model  = VADNet().to(self.device)
            self.model.load_state_dict(torch.load(CUSTOM_VAD_MODEL_PATH, map_location=self.device))
            self.model.eval()
            print("VAD: using custom neural model")
        else:
            self._use_fallback = True
            import webrtcvad
            self._fallback = webrtcvad.Vad(VAD_AGGRESSIVENESS)
            print("VAD: custom_vad.pt not found, falling back to webrtcvad")

    def _predict(self, frame: np.ndarray) -> int:
        features = extract_features(frame)
        tensor   = torch.tensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return logits.argmax(dim=1).item()

    def is_speech(self, frame_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        if self._use_fallback:
            return self._fallback.is_speech(frame_bytes, sample_rate)
        frame = np.frombuffer(frame_bytes, dtype=np.int16)
        return self._predict(frame) in ACTIVE_CLASSES

    def get_class(self, frame_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> int:
        """Returns raw class int: 0=silence, 1=speech, 2=overlap, 3=vocalization."""
        if self._use_fallback:
            return 1 if self._fallback.is_speech(frame_bytes, sample_rate) else 0
        frame = np.frombuffer(frame_bytes, dtype=np.int16)
        return self._predict(frame)
```

---

## Step 9 — `config.py` (MODIFIED)

Add after existing constants:
```python
CUSTOM_VAD_MODEL_PATH = "models/custom_vad.pt"
```

---

## Step 10 — Install Dependencies

```bash
pip install scikit-learn pandas soundfile tqdm
```

Everything else (`torch`, `librosa`, `webrtcvad`) you already have.

---

## Summary — Order of Operations

| # | Command | What it does |
|---|---------|-------------|
| 1 | Download datasets | LibriSpeech, AMI, ESC-50 (see Step 1) |
| 2 | `python -m ml.labeling.label_librispeech` | Labels ~2hrs of speech/silence |
| 3 | `python -m ml.labeling.label_ami` | Labels overlap + vocalization from meetings |
| 4 | `python -m ml.labeling.label_esc50` | Labels laughter + coughing clips |
| 5 | `python -m ml.merge_labels` | Combines all CSVs into `all_labels.csv` |
| 6 | `python -m ml.train` | Trains model, saves `models/custom_vad.pt` |
| 7 | `python -m ml.evaluate` | Checks per-class accuracy + confusion matrix |
| 8 | `python app.py` | App runs with custom VAD automatically |

The app falls back to `webrtcvad` automatically if `custom_vad.pt` doesn't exist yet,
so you can keep using the app normally while training.