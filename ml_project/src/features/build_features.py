import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from enities.feature_params import FeatureParams


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, params: FeatureParams):
        self.params = params
        self.categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown='ignore')),
            ]
        )
        self.numerical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

    def fit(self, X: pd.DataFrame, y=None):
        self.numerical_pipeline.fit(X[self.params.numerical_features])
        self.categorical_pipeline.fit(X[self.params.categorical_features])
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X_ = X.copy()
        new_categorical_cols = self.categorical_pipeline.get_params()['ohe'].\
                               get_feature_names(self.params.categorical_features)
        X_[self.params.numerical_features] = self.numerical_pipeline.transform(X_[self.params.numerical_features])
        X_[new_categorical_cols] = self.categorical_pipeline.transform(X_[self.params.categorical_features]).toarray()
        X_.drop(self.params.categorical_features, axis=1, inplace=True)
        return X_


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    if params.use_log_trick:
        target = pd.Series(np.log(target.to_numpy()))
    return target