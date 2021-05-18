import pytest

import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.data.data_validation import validate_data, validate_request
from src.app import UCIModel, app

DEFAULT_DATA_PATH = 'data/heart.csv'
client = TestClient(app)


@pytest.fixture(scope='session')
def etalon_request():
    data_to_test = pd.read_csv(DEFAULT_DATA_PATH)
    request_features = list(data_to_test.columns)
    request_features.remove('target')
    request_data = [
        x.item() if isinstance(x, np.generic) else x for x in data_to_test.iloc[0].tolist()
    ]
    request_data = request_data[:-1]
    return UCIModel(data=[request_data], features=request_features)


def test_validate_data_return_true_on_etalon_data():
    data_to_test = pd.read_csv(DEFAULT_DATA_PATH)
    output = validate_data(data_to_test)
    assert output


def test_validate_data_return_false_on_incorrect_data():
    incorrect_data = pd.DataFrame([1], columns=['something'])
    output = validate_data(incorrect_data)
    assert not output


def test_validate_request_return_true_on_etalon_request(etalon_request):
    output = validate_request(etalon_request)
    assert output


def test_service_request_return_200_on_etalon_request(etalon_request):
    response = client.get('/predict/', json={'data': etalon_request.data, 'features': etalon_request.features})
    assert response.status_code == 200


def test_service_request_return_400_on_bad_request():
    response = client.get('/predict/', json={'data': [[1, 2]], 'features': ['some', 'thing']})
    assert response.status_code == 400
