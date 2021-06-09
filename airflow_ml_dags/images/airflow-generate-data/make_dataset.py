from sklearn.datasets import make_blobs
import pandas as pd
import click
import os


@click.command('generate_data')
@click.argument('output_dir')
def generate_data(output_dir: str):
    X, y = make_blobs(n_samples=500, n_features=5, centers=2, random_state=17)
    print(os.path.realpath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X).to_csv(os.path.join(output_dir, 'data.csv'), index=False)
    pd.DataFrame(y.reshape(-1, 1)).to_csv(os.path.join(output_dir, 'target.csv'), index=False)


if __name__ == "__main__":
    generate_data()
