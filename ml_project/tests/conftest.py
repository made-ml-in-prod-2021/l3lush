import os
import pytest
from typing import List
import pandas as pd

from .generate_data import independent_generating

DEFAULT_FAKE_DATA_PATH = 'fake_data.csv'


@pytest.fixture()
def real_dataset_path():
    return 'data/raw/heart.csv'


@pytest.fixture()
def dataset_path(real_dataset_path: str):
    real_data = pd.read_csv(real_dataset_path)
    generated_data = independent_generating(real_data, size=1000)
    curdir = os.path.dirname(__file__)
    fake_data_path = os.path.join(curdir, DEFAULT_FAKE_DATA_PATH)
    generated_data.to_csv(fake_data_path, index=False)
    return fake_data_path


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
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


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]
