import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("data/test.wav", sr=16000)
S = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S), sr=sr)
plt.colorbar()
plt.show()