import os
import pytest
from typing import List
import pandas as pd

from .generate_data import independent_generating
from enities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params

DEFAULT_FAKE_DATA_PATH = 'fake_data.csv'


@pytest.fixture(scope='session')
def real_dataset_path():
    return 'data/raw/heart.csv'


@pytest.fixture(scope='session')
def dataset_path(real_dataset_path: str):
    real_data = pd.read_csv(real_dataset_path)
    generated_data = independent_generating(real_data, size=1000)
    curdir = os.path.dirname(__file__)
    fake_data_path = os.path.join(curdir, DEFAULT_FAKE_DATA_PATH)
    generated_data.to_csv(fake_data_path, index=False)
    return fake_data_path


@pytest.fixture(scope='session')
def target_col():
    return "target"


@pytest.fixture(scope='session')
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]


@pytest.fixture(scope='session')
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]


@pytest.fixture(scope='session')
def path_for_test_report_output() -> str:
    return os.path.join('models', 'metrics_from_test.json')


@pytest.fixture(scope='session')
def path_for_test_model_output() -> str:
    return os.path.join('models', 'model_from_test.pkl')


@pytest.fixture(scope='session')
def common_training_params() -> TrainingPipelineParams:
    params = read_training_pipeline_params('configs/train_config.yaml')
    return params


@pytest.fixture(scope='session')
def train_params(
        real_dataset_path, target_col, categorical_features, numerical_features,
        path_for_test_report_output, path_for_test_model_output,
        common_training_params
) -> TrainingPipelineParams:
    params = TrainingPipelineParams(
        input_data_path=real_dataset_path,
        output_model_path=path_for_test_model_output,
        metric_path=path_for_test_report_output,
        splitting_params=common_training_params.splitting_params,
        feature_params=common_training_params.feature_params,
        train_params=common_training_params.train_params
    )
    return params
