import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier

from config import SAMPLE_RATE

MODELS_DIR = "models"

class SpeakerEncoder:
    def __init__(self):
        self.classifier = EncoderClassifier.from_hparams( 
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=MODELS_DIR
        )
    
    def encode(self, audio_np, sample_rate=SAMPLE_RATE) -> np.ndarray:
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32) / 32768.0 # [-1, 1]
        
        tensor = torch.tensor(audio_np).unsqueeze(0) # N samples to [1, N]

        with torch.no_grad(): # no record gradient (intermediate) values
            embedding = self.classifier.encode_batch(tensor)
        
        return embedding.squeeze().cpu().numpy()