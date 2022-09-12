"""Additional metrics for this task."""
from scipy.spatial.distance import cosine
import numpy as np


def cosine_distance(y_true, y_pred):
    loss = []
    for y_t, y_p in zip(y_true, y_pred):
        loss.append(cosine(y_t, y_p))
    return np.mean(loss)
