import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from embeddings.encoder import SpeakerEncoder
from config import SAMPLE_RATE


def test_embedding_interface():
    encoder = SpeakerEncoder()

    # generate random fake audio
    audio1 = np.random.randn(SAMPLE_RATE).astype(np.float32)
    audio2 = np.random.randn(SAMPLE_RATE).astype(np.float32)

    embedding1 = encoder.encode(audio1, SAMPLE_RATE)
    embedding2 = encoder.encode(audio2, SAMPLE_RATE)

    # shape checks
    assert embedding1.shape == (192,)
    assert embedding2.shape == (192,)

    # dtype checks
    assert embedding1.dtype == np.float32
    assert embedding2.dtype == np.float32