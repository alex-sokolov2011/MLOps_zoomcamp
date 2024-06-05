import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import json

import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

EXPERIMENT_NAME = "hw3-exp-yellow-taxi-linreg"
MLFLOW_TRACKING_URI = "http://mlflow:5000"

@data_exporter
def export_data(data, *args, **kwargs):
    """
    """
    
    model, parameters, X, y, model_info, dv = data

    # Set MLFlow tracking URI and experiment name
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        print(f'Starting experiment {EXPERIMENT_NAME}')
        # Log the model to MLFlow
        mlflow.sklearn.log_model(model, "linreg_default")

        # Save and log the artifact 
        path = 'dv.pkl'
        with open(path, 'wb') as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(path, "artifacts")
        print("Experiment finished")
    
    # Disable autologging to avoid potential issues
    mlflow.sklearn.autolog(disable=True)
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    last_runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["attributes.start_time DESC"]
    )

    # Register the last model
    last_runs_id = last_runs[0].info.run_id
    model_uri = f"runs:/{last_runs_id}/model"
    mlflow.register_model(model_uri=model_uri, name="linreg-default")

    # Get the model size in bytes
    json_string = last_runs[0].data.tags['mlflow.log-model.history']
    json_log_model= json.loads(json_string)
    model_size_bytes = int(([entry["model_size_bytes"] for entry in json_log_model])[0])

    return {"model_size_bytes": model_size_bytes}
