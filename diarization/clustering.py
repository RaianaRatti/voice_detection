import numpy as np
from scipy.spatial.distance import cosine
from config import CLUSTER_THRESHOLD

def assign_speaker(new_embedding, history) -> int: # history = [(embedding, speaker_id)]
    # if no speakers
    if (not history):
        return 0

    # group emebddings by speaker and compute per-speaker centroid
    speaker_embeddings = {}
    for embedding, speaker_id in history:
        if speaker_id not in speaker_embeddings:
            speaker_embeddings[speaker_id] = []
        speaker_embeddings[speaker_id].append(embedding)

    centroids = {
        speaker_id: np.mean(np.stack(embeddings), axis=0)
        for speaker_id, embeddings in speaker_embeddings.items()
    }

    # compute cosine distance between new_embedding and each row
    best_distance = None
    best_id = None

    for speaker_id, centroid in centroids.items():
        distance = cosine(centroid, new_embedding)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_id = speaker_id

    if best_distance <= CLUSTER_THRESHOLD:
        return best_id

    existing_ids = list(speaker_embeddings.keys())
    return max(existing_ids) + 1
    