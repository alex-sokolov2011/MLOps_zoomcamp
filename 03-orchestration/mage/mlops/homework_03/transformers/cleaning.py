from typing import Tuple

import pandas as pd

from mlops.utils.data_preparation.hw3_cleaning import clean
from mlops.utils.data_preparation.hw3_splitters import split

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = clean(df)
    df_train, df_val = split(
        df
    )

    return df, df_train, df_val