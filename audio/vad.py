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