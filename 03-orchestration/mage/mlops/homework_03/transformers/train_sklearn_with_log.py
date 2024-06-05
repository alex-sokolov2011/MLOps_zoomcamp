from typing import Callable, Dict, Tuple, Union

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression

from pandas import Series
from scipy.sparse._csr import csr_matrix

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def default_training(
    training_set: Dict[str, Union[Series, csr_matrix]],
    *args,
    **kwargs,
) -> Tuple[
    BaseEstimator,
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
    Callable[..., BaseEstimator],
]:

    X, y, _ = training_set['encoder']
    
    model_class = load_class(model_class_name)
    model, metrics, y_pred = train_model(
        model_class(),
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y
    )

    parameters = model.get_params()    
    intercept = model.intercept_ if hasattr(model, 'intercept_') else 'N/A'
    alpha = model.alpha if hasattr(model, 'alpha') else 'N/A'

    parameters['intercept'] = intercept
    parameters['alpha'] = alpha
    parameters['mse_train'] = metrics['mse']
    parameters['rmse_train'] = metrics['rmse']
    
    
    return model, parameters, X, y, dict(cls=model_class, name=model_class_name)