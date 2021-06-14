import click
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import pickle


@click.command("preprocessing")
@click.argument("raw_data_dir")
@click.argument("processed_data_dir")
def preprocess_data(raw_data_dir: str, processed_data_dir: str):
    features = pd.read_csv(os.path.join(raw_data_dir, 'data.csv'))
    target = pd.read_csv(os.path.join(raw_data_dir, 'target.csv'))
    scaler = StandardScaler()
    features_transformed = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    joined_data = pd.concat([features_transformed, target], axis=1)
    print(features_transformed.shape, target.shape, joined_data.shape)
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    joined_data.to_csv(os.path.join(processed_data_dir, 'processed_data.csv'), index=False)

    with open(os.path.join(processed_data_dir, 'scaler.pkl'), 'wb') as fin:
        pickle.dump(scaler, fin)


if __name__ == '__main__':
    preprocess_data()
