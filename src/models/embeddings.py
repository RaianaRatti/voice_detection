# Turn audio clip into list of numbers (embedding)

from pyannote.audio import Model, Inference

model = Model.from_pretrained("pyannote/embedding")
inference = Inference(model, window="whole")
embedding = inference("data/test.wav")
print(embedding.shape)  # something like (1, 512)
print(embedding)         # a list of numbers — your voice's "fingerprint"