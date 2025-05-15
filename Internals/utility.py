"""Module which contain utilities function for ML and DL projects."""

import numpy as np


def shuffle_arrays(*arrays: np.ndarray, random_state: int = None) -> list[np.ndarray]:
    """Shuffle multiple arrays in the same order each.

    Args:
        arrays: Arrays to shuffle. All has to have the same number of samples (first dimention).
        random_state: Controls the shuffling for reprodusible results.

    Raises:
        ValueError: If not arrays are provided.
        ValueError: If any input is  not a NumPy array.
        ValueError: If nput arrays have different number of samples.

    Returns:
        list[np.array]: A list of arrays shuffled in the same order each.
    """
    if not arrays:
        raise ValueError('At least one array must be provided.')
    
    if not all(isinstance(array, np.ndarray) for array in arrays):
        raise ValueError("All arrays must be instance of np.ndarray.")
    
    n_samples = arrays[0].shape[0]
    if not all(array.shape[0] == n_samples for array in arrays):
        raise ValueError("All arrays must have the same number of samples")
    
    rng = np.random.default_rng(seed=random_state) if random_state is not None else None
    indices = np.arange(n_samples)
    if rng:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    return [array[indices] for array in arrays]
