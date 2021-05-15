from src.data.make_dataset import read_data, split_train_val_data
from enities.split_params import SplittingParams


def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert len(data) > 10, (
        'The result of reading data is too short.'
    )
    assert target_col in data.columns, (
        f'target column {target_col} not in data columns.'
    )


def test_split_dataset(dataset_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=239, val_size=val_size,)
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10
