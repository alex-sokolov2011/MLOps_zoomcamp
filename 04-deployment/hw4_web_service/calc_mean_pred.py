import warnings
import click
import pickle
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

CATEGORICAL = ['PULocationID', 'DOLocationID']

@click.command()
@click.option(
    '--year',
    type=int,
    required=True,
    help='Year of the trip data'
)
@click.option(
    '--month',
    type=int,
    required=True,
    help='Month of the trip data'
)
def main(year, month):
    with open('../description_homework/model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file)

    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f'Mean predicted duration: {y_pred.mean()}')

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')

    return df

if __name__ == '__main__':
    main()
