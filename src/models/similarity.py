# Cosine similarity to compare embeddings (standard approach)

from sklearn.metrics.pairwise import cosine_similarity
from pyannote.audio import Model, Inference

model = Model.from_pretrained("pyannote/embedding")
inference = Inference(model, window="whole")

emb1 = inference("data/you_clip1.wav")
emb2 = inference("data/you_clip2.wav")
emb3 = inference("data/someone_else.wav")

print(cosine_similarity(emb1, emb2))  # should be high (~0.8–0.99)
print(cosine_similarity(emb1, emb3))  # should be lower (~0.1–0.5)