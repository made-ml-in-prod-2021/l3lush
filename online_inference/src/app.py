import logging
import sys
import pickle
from typing import List, Union
import click

import pandas as pd
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, conlist

from src.enities.app_params import read_app_params, AppParams
from src.features.build_features import CustomTransformer
from src.data.data_validation import validate_request

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def load_object(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


class UCIModel(BaseModel):
    data: List[conlist(Union[float, str, None])]
    features: List[str]


class TargetResponse(BaseModel):
    id: str
    target: float


def make_predict(
    data: List, features: List[str], params: AppParams
) -> List[TargetResponse]:
    data = pd.DataFrame(data, columns=features)
    logger.info(f'got input data with shape {data.shape}')
    ids = [int(x) for x in data.index]
    model = load_object(params.model)
    transformer: CustomTransformer = load_object(params.transformer)
    features_ = params.feature_params.numerical_features + params.feature_params.categorical_features
    X = data[features_]
    X = transformer.transform(X)
    predicts = model.predict(X)
    logger.info(f'output with shape: {X.shape}')
    return [
        TargetResponse(id=id_, target=float(price)) for id_, price in zip(ids, predicts)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


async def common_params(config_path: str = 'configs/app_config.yaml'):
    return read_app_params(config_path)


@app.get("/predict/", response_model=List[TargetResponse])
def predict(request: UCIModel, params: AppParams = Depends(common_params)):
    if not validate_request(request):
        raise HTTPException(status_code=400, detail='Wrong request format')
    logger.info(f'load with params {request}')
    return make_predict(request.data, request.features, params)


@click.command(name="train_pipeline")
@click.argument("config_path")
def online_inference_app(config_path: str):
    if config_path == '':
        params = read_app_params('configs/app_config.yaml')
    else:
        params = read_app_params(config_path)
    logger.info(f'load with initial params {params}')
    uvicorn.run("app:app", host=params.host, port=params.port)


if __name__ == "__main__":
    online_inference_app()
