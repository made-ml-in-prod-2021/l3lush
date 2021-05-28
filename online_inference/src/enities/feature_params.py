from dataclasses import dataclass
from typing import List


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]