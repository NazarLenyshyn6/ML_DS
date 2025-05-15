"""Module which contain utilities function for ML and DL projects."""

import numpy as np


def shuffle_arrays(*arrays, random_state = None):
    if not all(isinstance(array, np.ndarray) for array in arrays):
        raise ValueError()
    
    n_samples = arrays[0].shape[0]
    if not all(array.shape[0] == n_samples for array in arrays):
        raise ValueError()
    
    rng = np.random.default_rng(seed=random_state) if random_state is not None else None
    indices = np.arange(n_samples)
    if rng:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    
    return [array[indices] for array in arrays]
