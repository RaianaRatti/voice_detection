# Process mic input in real-time with overlapping chunks, running diarization every few seconds

import queue
import threading
import sounddevice as sd
import numpy as np
import time

CHUNK = 5.0   # seconds per chunk
OVERLAP = 1.0 # seconds of overlap
SR = 16000


def start_stream(processor=None):
    buffer = np.zeros(int((CHUNK + OVERLAP) * SR)) # 40,000 audio samples
    q = queue.Queue()
    stop_event = threading.Event() # default false
    last_process_time = [time.time()]  # list so it's mutable inside nested function

    def callback(indata, frames, time_info, status): # indata is newest chunk of audio samples
        nonlocal buffer
        new_audio = indata[:, 0] # 1D array of audio samples
        buffer = np.roll(buffer, -len(new_audio)) # shifts old audio in buffer left by length of new audio, making room for new audio at end
        buffer[-len(new_audio):] = new_audio # appends new audio to end of buffer

        now = time.time()
        if processor is not None and (now - last_process_time[0]) >= CHUNK:
            last_process_time[0] = now
            q.put(processor(buffer.copy())) # calls processor function (run_diarization) on copy of buffer and puts result in q

    def wait_for_enter():
        input("Recording... press Enter to stop\n")
        stop_event.set()

    thread = threading.Thread(
        target=wait_for_enter, # thread will run this function (not immediately)
        daemon=True # Daemon thread killed at program end
    ) 
    thread.start() # runs wait_for_enter() in background

    # with allows automatic cleanup of audio stream on exit of block
    with sd.InputStream(samplerate=SR, channels=1, callback=callback): # whenever new audio arrives, call callback()
        while not stop_event.is_set():
            try:
                yield q.get(timeout=0.5) # wait max 0.5s for item
            except queue.Empty:
                continue


# START_STREAM()
# - Create zero array "buffer"
# - Create empty queue "q"
# - Create flag to stop "stop_event"
# - Create thread to wait for user to press Enter "thread"
# - Start audio stream with samplerate=SR, channels=1, and callback=callback (whenever new audio arrives, call callback())
# - While stop_event flag is not true
# - Obtain audio sample from q (wait 0.5s max)
# - If no sample, continue loop

# CALLBACK()
# - 

# WAIT_FOR_ENTER()
# - Prints "Recording... press Enter to stop"
# - Sets "stop_event" flag to true when user presses Enter

# NOTES FOR ME
# - "yield" allows multiple return values of the function
# - "with" allows code inside / infront of it to finish running, and cleans it up immediately (like opening file)