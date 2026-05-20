import sounddevice as sd
import numpy as np

from config import SAMPLE_RATE, FRAME_MS

def audio_capture(sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS):
    block_size = sample_rate * frame_ms // 1000
    with sd.InputStream(
        samplerate = sample_rate,
        channels = 1,
        dtype = 'int16',
        blocksize = block_size
    ) as stream:
        
        while True:
            audio_chunk, overflowed = stream.read(block_size)
            audio_chunk = np.squeeze(audio_chunk, axis=1)
            yield audio_chunk