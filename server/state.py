# Goal: pipeline thread writes and flask thread reads at same time, SharedState provides thread locking safety

import threading

class SharedState:
    def __init__(self):
        self.lock = threading.Lock() # ensures only one thread can access a specific piece of code at a time
        self.state = {
            "current": None,
            "times": {} # dict of {speaker_id: float}
        }
        self.pending_reset = False

    def update(self, speaker_id, times_dict):
        with self.lock:
            self.state = {
                "current": speaker_id,
                "times": times_dict
            }
    def reset(self):
        with self.lock:
            self.state = {"current": None, "times": {}}
            self.pending_reset = True
    
    def consume_reset(self):
        with self.lock:
            if self.pending_reset:
                self.pending_reset = False
                return True
            return False

    def get(self):
        with self.lock:
            return self.state.copy()