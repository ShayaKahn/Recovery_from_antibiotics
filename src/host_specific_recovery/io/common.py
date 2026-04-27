import pandas as pd
import numpy as np
from typing import Optional

def transpose_numeric(df: pd.DataFrame, norm: bool = False) -> np.ndarray:
    if norm is True:
        return df.div(df.sum(axis=1), axis=0).to_numpy().T
    else:
        return df.to_numpy().T


def load_csv_df(path: str, index_col: Optional[str | int] = None, sep=',') -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col, sep=sep)


def list_to_numpy(arr: list) -> np.ndarray:
    return np.asarray(arr, dtype=float)
