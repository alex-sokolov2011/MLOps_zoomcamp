import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
HPO_EXPERIMENT_NAME = "rf-hyperopt"
EXPERIMENT_NAME = "rf-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(data_path, params):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        print(f'Starting experiment {EXPERIMENT_NAME}')
        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        mlflow.log_metric("test_rmse", test_rmse)

        # Log model
        mlflow.sklearn.log_model(rf, artifact_path="model")

        print("Experiment finished")

    # Disable autologging to avoid potential issues
    mlflow.sklearn.autolog(disable=True)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    best_run_id = best_run.info.run_id
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="rf-best-model")

if __name__ == '__main__':
    run_register_model()
