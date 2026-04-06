from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureIndexSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_indices: list[int] | tuple[int, ...]):
        self.selected_indices = selected_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        indices = list(self.selected_indices)
        if hasattr(X, "iloc"):
            return X.iloc[:, indices]
        return X[:, indices]
