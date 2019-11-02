import tensorflow as tf
import numpy as np


# l2 normalization
def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
