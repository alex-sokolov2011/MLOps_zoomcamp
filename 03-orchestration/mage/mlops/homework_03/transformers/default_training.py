from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.models.hw3_sklearn import load_class, train_model

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def default_training(
    training_set: Dict[str, Union[Series, csr_matrix]],
    model_class_name: str,
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