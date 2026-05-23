import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from diarization.clustering import assign_speaker
from config import SAMPLE_RATE

def same_embedding(embedding1, embedding2) -> bool:
    # near identical embeddings
    embedding1 = np.random.randn(192).astype(np.float32)
    embedding2 = embedding1 + np.random.normal(
        scale=0.001,
        size=192
    ).astype(np.float32)

    history = []
    speaker1 = assign_speaker(embedding1, history)

    history.append((embedding1, speaker1))
    speaker2 = assign_speaker(embedding2, history)

    assert speaker1 == speaker2


def different_embeddings(embedding1, embedding2) -> bool:
    # orthogonal embeddings
    embedding1 = np.array(
        [1] + [0] * 191,
        dtype=np.float32
    )

    embedding2 = np.array(
        [0, 1] + [0] * 190,
        dtype=np.float32
    )

    history = []
    speaker1 = assign_speaker(embedding1, history)

    history.append((embedding1, speaker1))
    speaker2 = assign_speaker(embedding2, history)

    assert speaker1 != speaker2