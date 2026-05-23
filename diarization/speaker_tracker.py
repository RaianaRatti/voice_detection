from pathlib import Path
import sys
from config import HISTORY_WINDOW

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from diarization.clustering import assign_speaker

class SpeakerTracker:
    def __init__(self):
        history = [] # list of (emebdding, speaker_id)
        speak_times = {} # dict of {speaker_id: float}
        current_speaker = None # int or None
    
    # calls clustering.assign_speaker() to update speak_times, history, current_speaker, and return speaker_id
    def update(self, embedding, duration_seconds) -> int:
        speaker_id = assign_speaker(embedding, self.history)

        if (speaker_id not in self.speak_times):
            self.speak_times[speaker_id] = 0.0
        
        # update
        self.speak_times[speaker_id] += duration_seconds
        self.history.append((embedding, speaker_id))
        self.current_speaker = speaker_id

        # enforce HISTORY_WINDOW cap
        if len(self.history) > HISTORY_WINDOW:
            self.history.pop(0)
        
        return speaker_id

    # returns {"current": int, "times": {str: float}} - JSON payload sent to browser
    def get_state(self) -> dict:
        return {
            "current": self.current_speaker,
            "times": {
                str(speaker_id): duration for speaker_id, duration in self.speak_times.items()
            }
        }