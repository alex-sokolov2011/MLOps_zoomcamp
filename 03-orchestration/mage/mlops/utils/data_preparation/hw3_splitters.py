from typing import List, Tuple, Union

from pandas import DataFrame, Index


def split(
    df: DataFrame,
    return_indexes: bool = False,
) -> Union[Tuple[DataFrame, DataFrame], Tuple[Index, Index]]:
    df_train = df
    df_val = df.iloc(0)

    if return_indexes:
        return df_train.index, df_val.index

    return df_train, df_val