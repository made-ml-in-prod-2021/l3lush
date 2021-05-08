import os
import pickle
from typing import List, Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestClassifier

from src.data.make_dataset import read_data
from enities import TrainingParams
from enities.feature_params import FeatureParams
from src.features.build_features import extract_target, CustomTransformer
from src.models.train_model import train_model
from src.models.serialize_model import serialize_model


@pytest.fixture
def features_and_target(
    dataset_path: str, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col="target",
    )
    data = read_data(dataset_path)
    transformer = CustomTransformer(params)
    transformer.fit(data)
    features = transformer.transform(data)
    target = extract_target(data, params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, RandomForestClassifier)


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    n_estimators = 10
    model = RandomForestClassifier(n_estimators=n_estimators)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)
