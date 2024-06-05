from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy
from sklearn.feature_extraction import DictVectorizer


def vectorize_features(
    training_set: pd.DataFrame,
    ) -> Tuple[scipy.sparse.csr_matrix, DictVectorizer]:
    dv = DictVectorizer()

    train_dicts = training_set.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    return X_train, dv