import pandas as pd
import numpy as np


def independent_generating(data: pd.DataFrame, size: int = 500) -> pd.DataFrame:
    output = pd.DataFrame()
    for feature in data.columns:
        value_counts = data[feature].value_counts(normalize=True)
        keys, values = value_counts.index, value_counts.values
        output[feature] = np.random.choice(keys, p=values, size=size)
    return output
