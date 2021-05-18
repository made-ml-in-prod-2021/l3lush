import sys
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.app import app, make_predict, TargetResponse
from src.enities.app_params import read_app_params, AppParams

client = TestClient(app)


@pytest.fixture(scope='session')
def common_params() -> AppParams:
    return read_app_params('configs/app_config.yaml')


@pytest.fixture(scope='session')
def sample_data():
    data = pd.read_csv("data/heart.csv")
    data.drop(['target'], axis=1, inplace=True)
    request_features = list(data.columns)
    request_data = [
        x.item() if isinstance(x, np.generic) else x for x in data.iloc[0].tolist()
    ]
    return {"data": [request_data], "features": request_features}


def test_read_main():
    response = client.get('/')
    assert response.status_code == 200


def test_predict_return_right_format():
    response = client.get('/predict/')
    assert response.status_code == 422


def test_predict_function(sample_data, common_params):
    sys.path.append('src')
    output = make_predict(sample_data['data'], sample_data['features'], common_params)
    assert isinstance(output, list)
    assert isinstance(output[-1], TargetResponse)
    assert output[0].id == '0'
    assert len(output) == 1


def test_predict(sample_data):
    response = client.get('/predict/', json=sample_data)
    assert response.status_code == 200, f'{response.json()}'
    assert isinstance(response.json(), list)
    assert response.json()[0]['id'] == '0'
