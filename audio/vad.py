import numpy as np
from config import SAMPLE_RATE, VAD_AGGRESSIVENESS

SILENCE_CLASS      = 0
SPEECH_CLASS       = 1
OVERLAP_CLASS      = 2
VOCALIZATION_CLASS = 3
ACTIVE_CLASSES     = {SPEECH_CLASS, OVERLAP_CLASS, VOCALIZATION_CLASS}


class VAD:
    def __init__(self):
        print("VAD: using simple amplitude-based detection (placeholder)")

    def is_speech(self, frame_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        frame = np.frombuffer(frame_bytes, dtype=np.int16)
        amplitude = np.abs(frame).mean()
        threshold = 500
        return amplitude > threshold

    def get_class(self, frame_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> int:
        """Returns raw class int: 0=silence, 1=speech, 2=overlap, 3=vocalization."""
        return 1 if self.is_speech(frame_bytes, sample_rate) else 0
