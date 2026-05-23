from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# -------------------------------------------

import numpy as np
import time

from audio.vad import VAD
from embeddings.encoder import SpeakerEncoder
from diarization.speaker_tracker import SpeakerTracker
from audio.capture import audio_capture
from config import SAMPLE_RATE, FRAME_MS, MIN_SPEECH_MS

def run(state):
    vad = VAD()
    encoder = SpeakerEncoder()
    tracker = SpeakerTracker()

    current_frames = []
    was_speaking = False
    total_silence_seconds = 0.0

    for frame in audio_capture():

        if not state.is_running():
            time.sleep(0.05)
            continue

        if state.consume_reset():
            tracker = SpeakerTracker()
            current_frames = []
            was_speaking = False
            total_silence_seconds = 0.0

        frame_bytes = frame.tobytes()
        is_speaking = vad.is_speech(frame_bytes, SAMPLE_RATE)

        if is_speaking:
            current_frames.append(frame)

        else:
            total_silence_seconds += FRAME_MS / 1000.0

            if was_speaking and current_frames:
                utterance = np.concatenate(current_frames)
                duration_seconds = len(utterance) / SAMPLE_RATE
                duration_ms = duration_seconds * 1000

                if duration_ms >= MIN_SPEECH_MS:
                    utterance = utterance.astype(np.float32) / 32768.0
                    embedding = encoder.encode(utterance, SAMPLE_RATE)
                    tracker.update(embedding, duration_seconds)

                current_frames = []

            state.update(
                tracker.current_speaker,
                tracker.get_state()["times"],
                total_silence_seconds
            )

        was_speaking = is_speaking