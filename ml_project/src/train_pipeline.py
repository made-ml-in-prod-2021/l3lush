import json
import logging
import sys
import pickle

import click
import pandas as pd

from src.data.make_dataset import read_data, split_train_val_data
from enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.features.build_features import extract_target, CustomTransformer
from src.models.train_model import train_model
from src.models.serialize_model import serialize_model
from src.models.predict_model import predict_model
from src.models.evaluate_model import evaluate_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    transformer = CustomTransformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = transformer.transform(train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    with open('models/transformer.pkl', 'wb') as fin:
        pickle.dump(transformer, fin)

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )
    val_features = transformer.transform(val_df)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    val_features_prepared = prepare_val_features_for_predict(
        train_features, val_features
    )

    logger.info(f"val_features.shape is {val_features_prepared.shape}")
    predicts = predict_model(
        model,
        val_features_prepared,
        training_pipeline_params.feature_params.use_log_trick,
    )

    metrics = evaluate_model(
        predicts,
        val_target,
        use_log_trick=training_pipeline_params.feature_params.use_log_trick,
    )

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)

    return path_to_model, metrics


def prepare_val_features_for_predict(
    train_features: pd.DataFrame, val_features: pd.DataFrame
):
    # small hack to work with categories
    train_features, val_features = train_features.align(
        val_features, join="left", axis=1
    )
    val_features = val_features.fillna(0)
    return val_features


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()