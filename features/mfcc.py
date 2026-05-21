import numpy as np
import librosa

from config import SAMPLE_RATE

# Extract MFCC features from raw audio
def extract(audio_np, sample_rate=SAMPLE_RATE) -> np.ndarray:

    # Ensure typs of audio_np are float32 or float64 (best for librosa)
    if (audio_np.dtype != np.float32 and audio_np.dtype != np.float64):
        audio_np = audio_np.astype(np.float32) / 32768.0
    
    mfcc = librosa.feature.mfcc(
        y=audio_np,
        sr=sample_rate,
        n_mfcc=40
    )

    return mfcc