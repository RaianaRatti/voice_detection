# Goal: pipeline thread writes and flask thread reads at same time, SharedState provides thread locking safety

import threading

class SharedState:
    def __init__(self):
        self.lock = threading.Lock() # ensures only one thread can access a specific piece of code at a time
        self.state = {
            "current": None,
            "times": {}, # dict of {speaker_id: float}
            "silence": 0.0
        }
        self.pending_reset = False,
        self.running = False

    def update(self, speaker_id, times_dict, silence_seconds=0.0):
        with self.lock:
            self.state = {
                "current": speaker_id,
                "times": times_dict,
                "silence": silence_seconds
            }
    def reset(self):
        with self.lock:
            self.state = {"current": None, "times": {}, "silence": 0.0}
            self.pending_reset = True
    
    def consume_reset(self):
        with self.lock:
            if self.pending_reset:
                self.pending_reset = False
                return True
            return False
        
    def start(self):
        with self.lock:
            self.running = True

    def stop(self):
        with self.lock:
            self.running = False

    def is_running(self):
        with self.lock:
            return self.running

    def get(self):
        with self.lock:
            return self.state.copy()