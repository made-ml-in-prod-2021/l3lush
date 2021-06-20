import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import click
import os
import pickle


@click.command('train_model')
@click.argument('processed_data_dir')
@click.argument('models_dir')
def train_model(processed_data_dir: str, models_dir: str):
    train = pd.read_csv(os.path.join(processed_data_dir, 'train.csv'))
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    forest = RandomForestClassifier(n_estimators=10)
    forest.fit(X_train, y_train)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    with open(os.path.join(models_dir, 'model.pkl'), 'wb') as fin:
        pickle.dump(forest, fin)


if __name__ == '__main__':
    train_model()
