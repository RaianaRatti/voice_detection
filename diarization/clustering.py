import numpy as np
from scipy.spatial.distance import cosine
from config import CLUSTER_THRESHOLD

def assign_speaker(new_embedding, history) -> int: # history = [(embedding, speaker_id)]
    # if no speakers
    if (not history):
        return 0

    # compute cosine distance between new_embedding and each row
    best_distance = None
    best_id = None

    for current_embedding, current_id in history:
        distance = cosine(current_embedding, new_embedding)

        if (best_distance is None or distance < best_distance):
            best_distance = distance
            best_id = current_id
    
    if (best_distance <= CLUSTER_THRESHOLD):
        return best_id
    
    # new speaker
    existing_ids = [speaker_id for _, speaker_id in history]
    return max(existing_ids) + 1
    