# Labels each audio frame as speech or silence

import webrtcvad
import numpy as np

from config import SAMPLE_RATE, FRAME_MS, MIN_SPEECH_MS, VAD_AGGRESSIVENESS

class VAD:
    def __init__(self, aggressiveness=VAD_AGGRESSIVENESS):
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame_bytes, sample_rate=SAMPLE_RATE):
        return self.vad.is_speech(frame_bytes, sample_rate)
    
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