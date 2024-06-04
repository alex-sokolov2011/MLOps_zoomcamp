from typing import List, Tuple

from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.data_preparation.hw3_encoders import vectorize_features
from mlops.utils.data_preparation.hw3_feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
def export(
    df: DataFrame, *args, **kwargs
) -> Tuple[
    csr_matrix,
    Series,
    BaseEstimator,
]:
    target = kwargs.get('target')

    y: Series = df[target]

    X_train, dv = vectorize_features(
        select_features(df)
    )
    
    
    return X_train, y, dv


@test
def test_dataset(
    X_train: csr_matrix,
    y: Series,
    *args,
) -> None:
    assert (
        X_train.shape[0] == 3316216
    ), f'Entire dataset should have 3316216 examples, but has {X_train.shape[0]}'
    assert (
        X_train.shape[1] == 518
    ), f'Entire dataset should have 518 features, but has {X_train.shape[1]}'
    assert (
        len(y.index) == X_train.shape[0]
    ), f'Entire dataset should have {X.shape[0]} examples, but has {len(y.index)}'


