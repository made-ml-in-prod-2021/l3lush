from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml

from .feature_params import FeatureParams


@dataclass()
class AppParams:
    host: str
    port: int
    model: str
    transformer: str
    feature_params: FeatureParams


AppParamsSchema = class_schema(AppParams)


def read_app_params(path: str) -> AppParams:
    with open(path, 'r') as input_stream:
        schema = AppParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
