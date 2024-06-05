from typing import List, Optional

import pandas as pd

CATEGORICAL_FEATURES = ['PULocationID', 'DOLocationID']


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = CATEGORICAL_FEATURES 

    return df[columns]
