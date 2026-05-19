import sounddevice as sd
import scipy.io.wavfile as wav

def record_clip(filename="data/test.wav", duration=3, sample_rate=16000):
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    wav.write(filename, sample_rate, audio)
    return filename