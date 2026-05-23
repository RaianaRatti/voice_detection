from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# -------------------------------------------

import numpy as np

from audio.vad import VAD
from embeddings.encoder import SpeakerEncoder
from diarization.speaker_tracker import SpeakerTracker
from audio.capture import audio_capture
from config import SAMPLE_RATE

def run(state):
    vad = VAD()
    encoder = SpeakerEncoder()
    tracker = SpeakerTracker()

    current_frames = []

    was_speaking = False

    for frame in audio_capture():
        frame_bytes = frame.tobytes()
        is_speaking = vad.is_speech(frame_bytes, SAMPLE_RATE)

        if is_speaking:
            current_frames.append(frame)
        
        # end of utterance - process frames
        elif was_speaking and current_frames:
            utterance = np.concatenate(current_frames)
            duration_seconds = len(utterance) / SAMPLE_RATE

            utterance = utterance.astype(np.float32) / 32768.0

            embedding = encoder.encode(utterance, SAMPLE_RATE)
            speaker_id = tracker.update(embedding, duration_seconds)

            state.update(speaker_id, tracker.get_state()["times"]) # update state

        current_frames = [] # clean buffer
        was_speaking = is_speaking