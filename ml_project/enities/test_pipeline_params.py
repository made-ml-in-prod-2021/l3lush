from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml

from .feature_params import FeatureParams


@dataclass()
class TestPipelineParams:
    input_data_path: str
    input_model_path: str
    output_predict_path: str
    feature_params: FeatureParams


TestPipelineParamsSchema = class_schema(TestPipelineParams)


def read_test_pipeline_params(path: str) -> TestPipelineParams:
    with open(path, "r") as input_stream:
        schema = TestPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
