import os

from enities.train_pipeline_params import TrainingPipelineParams
from src.train_pipeline import train_pipeline


def test_full_pipeline_works_well(train_params: TrainingPipelineParams):
    train_pipeline(train_params)
    assert os.path.exists(train_params.metric_path)
    assert os.path.exists(train_params.output_model_path)
    os.remove(train_params.metric_path)
    os.remove(train_params.output_model_path)
    assert not os.path.exists(train_params.metric_path)
    assert not os.path.exists(train_params.output_model_path)
