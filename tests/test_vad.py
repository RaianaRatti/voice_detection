import numpy as np
from scipy.io import wavfile
from pathlib import Path

from audio.vad import VAD

# global
vad = VAD()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# obtain audio
_, empty_audio = wavfile.read(DATA_DIR / 'empty_audio.wav')
_, sine_audio = wavfile.read(DATA_DIR / 'sine_audio.wav')

# obtaining results
print("Empty Audio")
for i, (_, is_speech) in enumerate(vad.label_frames(empty_audio)):
    print(f"{is_speech}", end=" ")

print("\n\nSample Audio")
for i, (_, is_speech) in enumerate(vad.label_frames(sine_audio)):
    print(f"{is_speech}", end=" ")