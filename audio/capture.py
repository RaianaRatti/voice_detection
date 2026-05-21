import sounddevice as sd
import numpy as np

from config import SAMPLE_RATE, FRAME_MS

def audio_capture(sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS):
    block_size = sample_rate * frame_ms // 1000 # number of samples per audio chunk

    # since no audio source given, uses default (device mic)
    with sd.InputStream(
        samplerate = sample_rate,
        channels = 1,
        dtype = 'int16',
        blocksize = block_size
    ) as stream: # returns numpy.ndarray of int16 (defined in dtype)
        
        while True:
            audio_chunk, overflowed = stream.read(block_size)
            audio_chunk = np.squeeze(audio_chunk, axis=1)
            yield audio_chunk