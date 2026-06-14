import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path

from config import SAMPLE_RATE, FRAME_MS

FRAME_SIZE  = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
N_MFCC = 40
LABEL_MAP = {"silence": 0, "speech": 1, "overlap": 2, "vocalization": 3}

# Extract MFCC + delta + delta-delta + energy from one 30ms frame
# Converts raw audio into numerical features
def extract_features(frame: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if frame.dtype != np.float32:
        frame = frame.astype(np.float32) / 32768.0 # [-1.0, 1,0]

    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=N_MFCC, n_fft=480, hop_length=160)
    delta = librosa.feature.delta(mfcc, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
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
        self.df        = pd.read_csv(label_csv)
        self.audio_dir = Path(audio_dir)
        self.augment   = augment
        self.samples   = []
        self._load()

    # populates samples
    def _load(self):
        # group all rows with same entry under "filename" (group all frames part of same file) and iterate through them
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

    # adds some random noise / volume to frame (label stays same)
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

    # returns features + label given an index (part of the audio you want to work with)
    def __getitem__(self, idx):
        frame, label = self.samples[idx]
        if self.augment:
            frame = self._augment(frame)
        features = extract_features(frame)
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)