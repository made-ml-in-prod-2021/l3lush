import logging
import sys

import click
import pandas as pd

from src.data.make_dataset import read_data
from enities.predict_pipeline_params import (
    PredictPipelineParams,
    read_predict_pipeline_params
)
from src.features.build_features import CustomTransformer
from src.models.predict_model import predict_model
from src.models.load_model import load_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(predict_pipeline_params: PredictPipelineParams):
    logger.info(f"start test pipeline with params {predict_pipeline_params}")
    data = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    transformer = CustomTransformer(predict_pipeline_params.feature_params)
    transformer.fit(data)
    test_features = transformer.transform(data)

    logger.info(f"test_features.shape is {test_features.shape}")
    model = load_model(predict_pipeline_params.input_model_path)

    predicts = predict_model(
        model,
        test_features,
        predict_pipeline_params.feature_params.use_log_trick,
    )
    predicts = pd.DataFrame(predicts, columns=['predict'])
    predicts.to_csv(predict_pipeline_params.output_predict_path, index=False)
    logger.info(f'prediction available at path: {predict_pipeline_params.output_predict_path}')


@click.command(name="test_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    params = read_predict_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()