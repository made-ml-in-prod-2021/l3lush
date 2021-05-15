import pickle

from .train_model import SklearnRegressionModel


def load_model(model_path: str) -> SklearnRegressionModel:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
