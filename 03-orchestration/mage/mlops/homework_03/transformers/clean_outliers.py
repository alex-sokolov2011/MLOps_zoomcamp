import pandas as pd

from mlops.utils.data_preparation.hw3_cleaning import clean

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(df: pd.DataFrame) -> pd.DataFrame:

    df = clean(df)
    print(len(df))

    return df