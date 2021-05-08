from typing import List

import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.data.make_dataset import read_data
from enities.feature_params import FeatureParams
from src.features.build_features import extract_target, CustomTransformer


@pytest.fixture(scope='session')
def feature_params(
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
        use_log_trick=False
    )
    return params


def test_make_features(
    feature_params: FeatureParams, dataset_path: str,
):
    data = read_data(dataset_path)
    transformer = CustomTransformer(feature_params)
    transformer.fit(data)
    features = transformer.transform(data)
    assert not pd.isnull(features).any().any()


def test_extract_features(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)

    target = extract_target(data, feature_params)
    assert_allclose(
        data[feature_params.target_col].to_numpy(), target.to_numpy()
    )


def test_custom_transformer_works_right():
    params_custom = FeatureParams(
        categorical_features=['sex'],
        numerical_features=['score'],
        target_col='feature'
    )
    sample_data = pd.DataFrame([['Female', 100], ['Male', 2]], columns=['sex', 'score'])
    transformer = CustomTransformer(params_custom)
    transformer.fit(sample_data)
    transformed_data = transformer.transform(sample_data)
    print(transformed_data)
    expected_scaled_data = pd.DataFrame([[1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]],
                                        columns=['score', 'sex_Female', 'sex_Male'])

    pd.testing.assert_frame_equal(transformed_data, expected_scaled_data)
