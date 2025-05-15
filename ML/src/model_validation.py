"""Module on which will be implemeted basic tools for model validation."""

from typing import Callable

import numpy as np

from Internals.utility import shuffle_arrays


def train_test_split(
        *sets, 
        train_size: float = None, 
        test_size: float = None, 
        shuffle: bool = True, 
        random_state: int = None
        ) -> list[np.ndarray]:
    """Split input sets into train and test subsets in the specified ratio.

    Args:
        sets: Arrays to be split. All has to have the same number of samples (first dimention).
        train_size: Proportion of dataset to include in train split (between 0 and 1).
        test_size: Proportion of dataset to included in test split (between 0 and 1).
        shuffle: Whether to shuffle before splitting. Default True.
        random_state: Controls the shuffling for reprodusible results.

    Raises:
        ValueError: If no input sets are provided.
        ValueError: If neither train_size nor test_size is specified.
        ValueError: If train_size not between 0 and 1.
        ValueError: If input sets have different number of samples.

    Returns:
        list[np.ndarray]: A list containing train and test splits for each input set, in the same order.
                          For example: [X_train, X_test, Y_train, Y_test, ...]
    """
    if not sets:
        raise ValueError('At least one set must be provided.')
    
    if not train_size and not test_size:
        raise ValueError("Either train_size or test_size must be specified.")
    
    n_sampels = sets[0].shape[0]
    if not all(set.shape[0] == n_sampels for set in sets):
        raise ValueError(f"All input sets must have the same number of samples. Got instead: {[set.shape[0] for set in sets]}")
    
    train_size =  train_size if train_size else 1.0 - test_size
    if not 0 < train_size < 1:
        raise ValueError(f"train_size must be between 0 and 1. Got: {train_size}")
    
    train_threshold = int(train_size * n_sampels)
    splits = []
    if shuffle:
        sets = shuffle_arrays(*sets, random_state=random_state)
        
    for set in sets:
        splits.append(set[:train_threshold])
        splits.append(set[train_threshold:])
    return splits


def cross_validate(
        X: np.ndarray, 
        Y: np.ndarray, 
        model, 
        loss_fn: Callable, 
        n_splits: int = 3, 
        shuffle: bool = True, 
        random_state: int = None
        ) -> np.ndarray:
    """Cross validation model evaluation.

    Args:
        X: Features array.
        Y: Targets array.
        model: Model to validate.
        loss_fn: Loss function for model validation.
        n_splits: Number of validation splits.
        shuffle: Whether to shuffle before splitting. Default True.
        random_state: Controls the shuffling for reprodusible results.

    Raises:
        ValueError: If X or Y is not NumPy array.
        ValueError: If X or Y have different number of samples (fist dimention).

    Return:
        np.ndarray: Loss value for all validation splits.
    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise ValueError("Both X and Y must be NumPy arrays.")
    
    if not X.shape[0] == Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples (fist dimention).")
    
    fold_size = X.shape[0] // n_splits
    fold_start_idx, fold_end_idx = 0, fold_size
    validation_losses = []
    if shuffle:
        X, Y = shuffle_arrays(X, Y, random_state=random_state)

    for _ in range(n_splits):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[fold_start_idx: fold_end_idx] = False
        X_train, Y_train = X[mask], Y[mask]
        X_val, Y_val = X[fold_start_idx: fold_end_idx], Y[fold_start_idx: fold_end_idx]
        model.fit(X=X_train, y=Y_train)
        loss = loss_fn(Y_val, model.predict(X=X_val))
        validation_losses.append(loss)
        fold_start_idx += fold_size
        fold_end_idx += fold_size
    return np.array(validation_losses)
