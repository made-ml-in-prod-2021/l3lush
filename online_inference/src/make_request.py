import numpy as np
import pandas as pd
import requests


if __name__ == "__main__":
    data = pd.read_csv("data/heart.csv")
    data.drop(['target'], axis=1, inplace=True)
    request_features = list(data.columns)
    for i in range(100):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_data)
        print(request_features)
        response = requests.get(
            "http://127.0.0.1:8000/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())