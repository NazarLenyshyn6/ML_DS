"""Module which contains functionality for imputing missing values."""

from typing import ClassVar, Any, Literal, Union, Optional

import numpy as np
import pandas as pd
import pydantic

from typing import ClassVar, Any, Literal, Union, Optional

import pydantic

class SimpleImputer(pydantic.BaseModel):
    """Impute missing values in a DataFrame using a specified strategy.

    Attributes:
        missing_values (Any): Value to treat as missing. Defaults to np.nan.
        strategy (str): One of {"mean", "median", "most_frequent", "constant"}.
        fill_value (Any): Value to use for filling when strategy == "constant".
        inplace (bool): Whether to modify the DataFrame in place.
    """

    _strategies: ClassVar = {
        'mean': lambda X, fill_value: np.nanmean(X, axis=0),
        'median': lambda X, fill_value: np.nanmedian(X, axis=0),
        'most_frequent': lambda X, fill_value: X.mode(0).iloc[0].values,
        'constant': lambda X, fill_value: np.full(X.shape[1], fill_value)
        }

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    missing_values: Union[np.number, int, float, str, Any] = pydantic.Field(default=np.nan)
    strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = pydantic.Field(default='mean')
    fill_value: Optional[Any] = pydantic.Field(default=0)
    inplace: bool = pydantic.Field(default=False)
    statistics_: np.ndarray = pydantic.Field(default=None, repr=False)
    fit_features: pd.Index = pydantic.Field(default=None, repr=False)
    fit_dtypes: np.ndarray = pydantic.Field(default=None, repr=False)

    def _validate_input_type(self, X: pd.DataFrame) -> None:
        if self.strategy in {'mean', 'median'}:
            if not np.issubdtype(X.values.dtype, np.number):
                raise TypeError(f'While using {self.strategy} impute strategy all values is DataFrame has to be numbers.')
            
    def _validate_input_features(self, X: pd.DataFrame) -> None:
            if X.columns.shape != self.fit_features.shape:
                raise ValueError('Number of features in input DataFrame must be equal to the number of features seen during fitting.')
            if not X.columns.equals(self.fit_features):
                raise ValueError(f"Features is input DataFrame must be equal to the feature seen during fitting.")
            if not all(X.dtypes.values == self.fit_dtypes):
                raise ValueError(f"Features data types in input DataFrame must be equal to features data types seen during fitting.")
            
    def  _check_is_fitted(self):
        if self.statistics_ is None:
            raise RuntimeError('SimpleImputer has to be fit. Call .fit() first.')
        
    def _find_nans(self, X: pd.DataFrame):
        if isinstance(self.missing_values, float) and np.isnan(self.missing_values):
            return np.where(X.isna())
        else:
            return np.where(X.values == self.missing_values)
        
    def _impute(self, X: pd.DataFrame, X_imputed: np.ndarray) -> pd.DataFrame:
        if self.inplace:
            X.iloc[:, :] = X_imputed
            return X
        else:
            return pd.DataFrame(X_imputed, columns=X.columns)
            
    def fit(self, X: pd.DataFrame) -> None:
        """Compute statistics to fill missing values.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Raises:
            TypeError: If using a numeric strategy on non-numeric data.
        """
        self._validate_input_type(X)
        self.statistics_ = self._strategies[self.strategy](X, self.fill_value)
        self.fit_features = X.columns
        self.fit_dtypes = X.dtypes.values

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using fitted statistics.

        Args:
            X (pd.DataFrame): DataFrame with potential missing values.

        Returns:
            pd.DataFrame: DataFrame with imputed values.

        Raises:
            RuntimeError: If called before fitting.
            ValueError: If feature mismatch occurs.
        """
        self._check_is_fitted()
        self._validate_input_features(X)
        X_imputed = X.values.astype(object) if not isinstance(self.fill_value, (int, float)) else X.values
        nan_idxs = self._find_nans(X)
        X_imputed[nan_idxs] = self.statistics_[nan_idxs[1]]
        return self._impute(X, X_imputed)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit imputer and transform the DataFrame in one step.

        Args:
            X (pd.DataFrame): DataFrame to fit and transform.

        Returns:
            pd.DataFrame: Imputed DataFrame.
        """
        self.fit(X)
        return self.transform(X)