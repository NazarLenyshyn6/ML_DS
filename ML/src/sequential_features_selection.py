"""Module containing implementation of sequential features selection."""

from typing import Literal, Optional

import numpy as np
import pandas as pd
import pydantic
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

from Internals.utility import check_dtype

# 
class SequentialFeatureSelection(pydantic.BaseModel):
    """Sequentily select features by taking on every step that one which leads to lowest loss.

    model: ML model to use for feature selection.
    n_features_to_select: Number of features to have in forward direction and number of feature to remove in backward direction.
    direction: Either to select feature from 0 to n (forward) or from n to 0 (backward).
    scoring: Function to evaluate loss.
    cv: Number of cross validation.
    early_stopping_treshold: Model improvement treshold before early stopping.

    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: BaseEstimator
    n_features_to_select: int | float
    direction: Literal['forward', 'backward'] = pydantic.Field(default='forward')
    scoring: str = pydantic.Field(default='r2')
    cv: int = pydantic.Field(default=5)
    n_features_in: int = pydantic.Field(default=None, init=False)
    feature_names_in: pd.Index = pydantic.Field(default=None, init=False)
    early_stopping_treshold: Optional[float] = pydantic.Field(default=None)

    def _check_if_fit(self):
        """Validate if SequentialFeatureSelection has been fit before calling transform or trying to get selected features.

        Raises:
            RuntimeError: If SequentialFeatureSelection has been call before fit.
        """
        if not hasattr(self, '_selected_features') or not hasattr(self, 'feature_names_in'):
            raise RuntimeError('SequentialFeatureSelection must be fit. Call .fit() first.')
        
    def get_support(self):
        """Return boolean mask array indicating selected features.

            Raises:
                RuntimeError: If trying to get selected features before SequentialFeatureSelection is fit.
                RuntimeError: If not feature impored the score;
        """
        self._check_if_fit()
        if self._selected_features is  None:
            raise RuntimeError('No feature impored the score; unable to proceed with selection.')
        return self._selected_features
    
    def get_features_names_out(self):
        """Return pd.Index of selected features names.
        
            Raises:
                RuntimeError: If trying to get selected features names before SequentialFeatureSelection is fit.
        """
        self._check_if_fit()
        return self.feature_names_in[self.get_support()]
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Peform bottom-up or top-down feature selection depending of spedified direction.

        Args:
            X: Features DataFrame.
            y: Target Series.
        
        Raises:
            TypeError: If X is not pandas DataFrame.
            TypeError: If y is not pandas Series.
            ValueError: If n_features_to_select specified as flaot number and not in range of (0, 1).
        """
        for validation_config in [(X, 'X', pd.DataFrame), (y, 'y', pd.Series)]:
            check_dtype(*validation_config)

        if isinstance(self.n_features_to_select, float):
            if not (0 < self.n_features_to_select < 1):
                raise ValueError('When n_features_to_select specified as float number it must be between 0 and 1.') 
            self.n_features_to_select = max(1, int(self.n_features_to_select*X.shape[1]))

        self.n_features_in = X.shape[1]
        self.feature_names_in = X.columns
        self._set_feature_names_in = set(self.feature_names_in)
        prev_score = 0.0

        self._selected_features = np.zeros(X.shape[1], dtype=bool) if self.direction == 'forward' else np.ones(X.shape[1], dtype=bool)
        avaliable_features_idx = set(range(X.shape[1]))
        for _ in range(self.n_features_to_select):
            max_score = -float('inf')
            selected_feature_idx = None
            for feature_idx in avaliable_features_idx:
                self._selected_features[feature_idx] = True if self.direction == 'forward' else False
                x = X.values[:, self._selected_features]
                score = cross_val_score(self.model, x, y, scoring=self.scoring, cv=self.cv).mean()
                if score >= max_score:
                    max_score = score
                    selected_feature_idx = feature_idx
                self._selected_features[feature_idx] = False if self.direction == 'forward' else True
            self._selected_features[selected_feature_idx] = True if self.direction == 'forward' else False
            avaliable_features_idx.remove(selected_feature_idx)
            if self.early_stopping_treshold is not None:
                if (max_score - prev_score) < self.early_stopping_treshold:
                    break
                prev_score = max_score

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame to DataFrame with features selected during fit.

        Args:
            X: Features DataFrame

        Raises:
            TypeError: If X is not DataFrame.
            ValueError: If number of feature in X is not equal to number of features seen during fitting.
            ValueError: If features names in X is not equal to features names seen during fitting.
        """
        check_dtype(X, 'X', pd.DataFrame)
        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f'Number of features in X must be equal to number of features seen during fitting ({self.n_features_in}). Got instead: {X.shape[1]}.'
                )
        if not all(feature in self._set_feature_names_in for feature in X.columns):
            raise ValueError(f'Features names is X must  be equal to feature names seen during fitting. Check feature_names_in.')
        
        return X.loc[:, self.get_features_names_out()]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit to the data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
           