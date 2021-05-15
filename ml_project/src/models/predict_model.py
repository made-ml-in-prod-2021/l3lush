import numpy as np
import pandas as pd

from .train_model import SklearnRegressionModel


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame, use_log_trick: bool = False
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts
