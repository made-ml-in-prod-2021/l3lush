import pandas as pd
import pickle

from src.enities.app_params import read_app_params, AppParams
from src.features.build_features import CustomTransformer

DEFAULT_TRAIN_DATA = 'data/heart.csv'


def train_transformer(config_path: str):
    params = read_app_params(config_path)
    transformer = CustomTransformer(params.feature_params)
    train_data = pd.read_csv(DEFAULT_TRAIN_DATA)
    transformer.fit(train_data)
    with open('models/saved_transformer.pkl', 'wb') as fin:
        pickle.dump(transformer, fin)


if __name__ == '__main__':
    train_transformer('configs/app_config.yaml')
