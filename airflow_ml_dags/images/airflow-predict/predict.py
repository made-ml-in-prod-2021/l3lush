import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import click
import pickle


@click.command('validate')
@click.command('processed_data_dir')
@click.command('models_dir')
def predict(processed_data_dir: str, models_dir: str):
    data_to_predict = pd.read_csv(os.path.join(processed_data_dir, 'test.csv'))
    if 'target' in data_to_predict.columns:
        X = data_to_predict.drop(['target'], axis=1)
    else:
        X = data_to_predict.copy()

    with open(os.path.join(processed_data_dir, 'scaler.pkl'), 'rb') as fin:
        scaler = pickle.load(fin)

    with open(os.path.join(models_dir, 'model.pkl'), 'rb') as fin:
        forest: RandomForestClassifier = pickle.load(fin)

    X = scaler.transform(X)
    y_pred = forest.predict(X)
    y_pred.to_csv(os.path.join(processed_data_dir, 'predictions.csv'), index=False)


if __name__ == "__main__":
    predict()