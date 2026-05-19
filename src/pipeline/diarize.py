# Take multi-speaker audio and segment into person speaking

import numpy as np
import torch
import os
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# Load .env from project root regardless of where Python is run from
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Load model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"])

if pipeline is None:
    raise RuntimeError("Failed to load pyannote pipeline. Check your token and that you've accepted the model terms at hf.co/pyannote/speaker-diarization-3.1")

def run_diarization(audio_source):
    if isinstance(audio_source, str):
        diarization = pipeline(audio_source)

    elif isinstance(audio_source, np.ndarray):
        # Convert to the format pyannote expects
        tensor = torch.tensor(audio_source).float().unsqueeze(0)  # shape: (1, samples)
        audio_dict = {"waveform": tensor, "sample_rate": 16000}
        diarization = pipeline(audio_dict)

    else:
        raise ValueError("run_diarization expects a file path or numpy array.")

    result = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        result[speaker] = result.get(speaker, 0) + duration

    return result