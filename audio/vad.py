# Labels each audio frame as speech or silence

import webrtcvad
import numpy as np

from config import SAMPLE_RATE, FRAME_MS, MIN_SPEECH_MS, VAD_AGGRESSIVENESS

class VAD:
    def __init__(self, aggressiveness=VAD_AGGRESSIVENESS):
        self.vad = webrtcvad.Vad(aggressiveness)

    # webrtc vad require frames to be passed as raw binary bytes (use .tobytes() to pass)
    def is_speech(self, frame_bytes, sample_rate=SAMPLE_RATE):
        return self.vad.is_speech(frame_bytes, sample_rate)

    def label_frames(self, audio, sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS):
        frame_size = int(sample_rate * frame_ms / 1000) # samples per frame (audio chunk)
        labeled_frames = []

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i+frame_size]

            # skip incomplete frames
            if len(frame) != frame_size:
                continue

            frame_bytes = frame.tobytes()
            speech = self.is_speech(frame_bytes, sample_rate)
            labeled_frames.append([frame, speech])

        return labeled_frames
    
    def merge_frames(self, frames, # list of (frame, is_speech)
                     sample_rate = SAMPLE_RATE,
                     frame_ms = FRAME_MS,
                     min_speech_ms = MIN_SPEECH_MS
                     ):
        
        utterances = []
        current_speech = []

        for frame, is_speech in frames:
            if is_speech:
                current_speech.append(frame)
            else:
                if current_speech:
                    utterance = np.concatenate(current_speech)
                    duration_ms = (len(utterance) / sample_rate) * 1000
                    
                    if duration_ms >= min_speech_ms:
                        utterances.append(utterance)
                    
                    current_speech = []
        
        # handling trailing speech
        if current_speech:
            utterance = np.concatenate(current_speech)
            duration_ms = (len(utterance) / sample_rate) * 1000
            
            if duration_ms >= min_speech_ms:
                utterances.append(utterance)
        
        return utterances