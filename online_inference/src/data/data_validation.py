import pandas as pd


def validate_data(data_to_validate: pd.DataFrame,
                  etalon_data_path: str='data/heart.csv'
    ):
    etalon_data = pd.read_csv(etalon_data_path)
    for i, column in enumerate(data_to_validate.columns):
        if etalon_data.columns[i] != column:
            return False
        if etalon_data.dtypes[column] != data_to_validate.dtypes[column]:
            return False
    return True


def validate_request(request_to_validate,
                     etalon_data_path: str='data/heart.csv'):
    etalon_df = pd.read_csv(etalon_data_path)
    etalon_df.drop(['target'], axis=1, inplace=True)
    etalon_features = etalon_df.columns.to_list()
    request_features = request_to_validate.features
    for i in range(len(request_features)):
        try:
            if request_features[i] != etalon_features[i]:
                print(f'WRONG columns: etalon {etalon_features[i]}, got {request_features[i]}')
                return False
        except:
            print('ERROR in column check')
            return False
    return True
