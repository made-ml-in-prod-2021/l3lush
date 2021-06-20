import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import click
import pickle
import json


@click.command('validate')
@click.argument('processed_data_dir')
@click.argument('models_dir')
def validate_model(processed_data_dir: str, models_dir: str):
    test = pd.read_csv(os.path.join(processed_data_dir, 'test.csv'))
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    with open(os.path.join(models_dir, 'model.pkl'), 'rb') as fin:
        forest: RandomForestClassifier = pickle.load(fin)

    y_pred_proba = forest.predict_proba(X_test)[:, 1]
    y_pred = forest.predict(X_test)

    metrics = {'accuracy': accuracy_score(y_test, y_pred),
               'precision': precision_score(y_test, y_pred),
               'roc-auc': roc_auc_score(y_test, y_pred_proba)}

    with open(os.path.join(models_dir, 'metrics.json'), 'w') as fin:
        json.dump(metrics, fin)


if __name__ == "__main__":
    validate_model()