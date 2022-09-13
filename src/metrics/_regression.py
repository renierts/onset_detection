"""Additional metrics for this task."""
from scipy.spatial.distance import cosine
import numpy as np


def cosine_distance(y_true, y_pred):
    """
    Compute the cosine distance between each target and prediction sequence.

    Parameters
    ----------
    y_true : np.ndarray, dtype=object, shape(n_sequences, )
        The targets. Each element of y_true is a numpy array of shape (
        n_samples, 1), where n_samples is the sequence  length.
    y_pred : np.ndarray, dtype=object, shape(n_sequences, )
        The predictions. Each element of y_pred is a numpy array of shape (
        n_samples, 1), where n_samples is the sequence  length.
    """
    loss = []
    for y_t, y_p in zip(y_true, y_pred):
        loss.append(cosine(y_t, y_p))
    return np.mean(loss)
