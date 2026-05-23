# Goal: pipeline thread writes and flask thread reads at same time, SharedState provides thread locking safety

import threading

class SharedState:
    def __init__(self):
        self.lock = threading.Lock() # ensures only one thread can access a specific piece of code at a time
        self.state = {
            "current": None,
            "times": {} # dict of {speaker_id: float}
        }

    def update(self, speaker_id, times_dict):
        with self.lock:
            self.state = {
                "current": speaker_id,
                "times": times_dict
            }

    def get(self):
        with self.lock:
            return self.state.copy()