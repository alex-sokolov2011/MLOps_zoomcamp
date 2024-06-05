import requests
from io import BytesIO
from typing import List
import gc

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

LOAD_COLUMNS = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    for year, months in [(2023, (3, 4))]:
        for i in range(*months):
            print(year, i)
            response = requests.get(
                'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_'
                f'{year}-{i:02d}.parquet'
            )

            if response.status_code != 200:
                raise Exception(response.text)

            df = pd.read_parquet(BytesIO(response.content), columns=LOAD_COLUMNS)
            dfs.append(df)
            del df
            gc.collect() 

    max_size_df = kwargs.get('max_size_df')
    if max_size_df > 0:
        df = pd.concat(dfs)[:max_size_df]
    else:
        df = pd.concat(dfs)

    print(len(df))

    return df