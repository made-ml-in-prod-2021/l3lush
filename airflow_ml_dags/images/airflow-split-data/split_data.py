import pandas as pd
from sklearn.model_selection import train_test_split
import os
import click


@click.command('split_data')
@click.argument('processed_data_dir')
def split_data(processed_data_dir: str):
    processed_data = pd.read_csv(os.path.join(processed_data_dir, 'processed_data.csv'))
    train, test = train_test_split(processed_data, test_size=0.3, random_state=17)
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    train.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    split_data()