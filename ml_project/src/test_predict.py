import logging
import sys

import click
import pandas as pd

from src.data.make_dataset import read_data
from enities.test_pipeline_params import (
    TestPipelineParams,
    read_test_pipeline_params
)
from src.features import make_features
from src.features.build_features import build_transformer
from src.models.predict_model import predict_model
from src.models.load_model import load_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def test_pipeline(test_pipeline_params: TestPipelineParams):
    logger.info(f"start test pipeline with params {test_pipeline_params}")
    data = read_data(test_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    transformer = build_transformer(test_pipeline_params.feature_params)
    transformer.fit(data)
    test_features = make_features(transformer, data)

    logger.info(f"test_features.shape is {test_features.shape}")
    model = load_model(test_pipeline_params.input_model_path)

    predicts = predict_model(
        model,
        test_features,
        test_pipeline_params.feature_params.use_log_trick,
    )
    predicts = pd.DataFrame(predicts, columns=['predict'])
    predicts.to_csv(test_pipeline_params.output_predict_path, index=False)
    logger.info(f'prediction available at path: {test_pipeline_params.output_predict_path}')


@click.command(name="test_pipeline")
@click.argument("config_path")
def test_pipeline_command(config_path: str):
    params = read_test_pipeline_params(config_path)
    test_pipeline(params)


if __name__ == "__main__":
    test_pipeline_command()