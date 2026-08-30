import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

TAXONOMIC_PREFIX = "taxonomic__"
FUNCTIONAL_PREFIX = "functional__"
METADATA_PREFIX = "metadata__"


def _is_binary_numeric_column(column):
    values = column.dropna().unique()
    return values.size > 0 and set(values).issubset({0, 1})


def _select_binary_numeric_columns(X):
    numeric = X.select_dtypes(include=np.number)
    return [column for column in numeric.columns if _is_binary_numeric_column(numeric[column])]


def _select_continuous_numeric_columns(X):
    numeric = X.select_dtypes(include=np.number)
    return [column for column in numeric.columns if not _is_binary_numeric_column(numeric[column])]


def combine_taxonomic_functional_tables(taxonomic_df, functional_df, metadata_df=None):
    """Combine taxonomic, functional, and optionally metadata tables.

    All supplied tables must be pandas DataFrames indexed by sample ID. The
    functional and metadata tables are aligned to the taxonomic table's sample
    order before combining. Columns are prefixed so that
    MicrobiomeRegularizedLinearRegressor and MicrobiomeRandomForestRegressor can
    route them to the matching preprocessing branch.

    Inputs:
    taxonomic_df: pandas.DataFrame of shape (n_samples, n_taxa), indexed by sample ID.
    functional_df: pandas.DataFrame of shape (n_samples, n_pathways), indexed by the same
                   sample IDs as taxonomic_df.
    metadata_df: optional pandas.DataFrame of shape (n_samples, n_metadata_features),
                 indexed by the same sample IDs as taxonomic_df. Numeric and categorical
                 columns are supported.

    Return:
    combined: pandas.DataFrame row-ordered like taxonomic_df, with
              "taxonomic__"/"functional__"/"metadata__" prefixed columns.
    """
    if not isinstance(taxonomic_df, pd.DataFrame) or not isinstance(functional_df, pd.DataFrame):
        raise TypeError("taxonomic_df and functional_df must be pandas DataFrames indexed by sample ID.")
    if metadata_df is not None and not isinstance(metadata_df, pd.DataFrame):
        raise TypeError("metadata_df must be a pandas DataFrame indexed by sample ID or None.")

    tables = [("taxonomic_df", taxonomic_df), ("functional_df", functional_df)]
    if metadata_df is not None:
        tables.append(("metadata_df", metadata_df))

    for table_name, table in tables:
        if table.index.duplicated().any():
            duplicates = table.index[table.index.duplicated()].unique().tolist()
            raise ValueError(f"{table_name} contains duplicated sample IDs: {duplicates}.")

    for table_name, table in tables[1:]:
        missing_from_table = taxonomic_df.index.difference(table.index).tolist()
        missing_from_taxonomic = table.index.difference(taxonomic_df.index).tolist()
        if missing_from_table or missing_from_taxonomic:
            raise ValueError(
                f"taxonomic_df and {table_name} must contain exactly the same sample IDs. "
                f"Missing from {table_name}: {missing_from_table}. "
                f"Missing from taxonomic_df: {missing_from_taxonomic}."
            )

    combined_tables = [
        taxonomic_df.add_prefix(TAXONOMIC_PREFIX),
        functional_df.loc[taxonomic_df.index].add_prefix(FUNCTIONAL_PREFIX),
    ]
    if metadata_df is not None:
        combined_tables.append(metadata_df.loc[taxonomic_df.index].add_prefix(METADATA_PREFIX))

    return pd.concat(combined_tables, axis=1)


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Remove taxa observed in fewer than `min_prevalence` fraction of samples.

    Parameters:
    min_prevalence : float
        Fraction in [0, 1] of samples where a taxon must be nonzero to be kept.

    Accepted input:
    X : pandas.DataFrame or numpy.ndarray
        2D abundance matrix with shape (n_samples, n_taxa). Values must be numeric and non-negative.
    """

    def __init__(self, min_prevalence=0.1):
        self.min_prevalence = min_prevalence

    def fit(self, X, y=None):
        if not isinstance(self.min_prevalence, (int, float)) or not (0 <= self.min_prevalence <= 1):
            raise ValueError("min_prevalence must be a number between 0 and 1.")

        X_arr = self._to_numpy(X)

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([f"taxon_{i}" for i in range(X_arr.shape[1])])

        prevalence = np.mean(X_arr > 0, axis=0)
        self.keep_mask_ = prevalence >= self.min_prevalence

        if not np.any(self.keep_mask_):
            raise ValueError("No taxa passed the prevalence filter.")

        return self

    def transform(self, X):
        X_arr = self._to_numpy(X)
        return X_arr[:, self.keep_mask_]

    def get_feature_names_out(self):
        return self.feature_names_in_[self.keep_mask_]

    @staticmethod
    def _to_numpy(X):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas.DataFrame or a 2D numpy.ndarray.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix: samples x taxa.")

        if not np.all(np.isfinite(X)):
            raise ValueError("Microbial abundance values must be finite and non-missing.")

        if np.any(X < 0):
            raise ValueError("Microbial abundance values must be non-negative.")

        return X


class AbundanceTransformer(BaseEstimator, TransformerMixin):
    """Transform microbial abundance values before model fitting.

    Parameters:
    method : str
        One of "none", "log", or "clr".
    pseudocount : float
        Positive value added before CLR transformation.
    """

    def __init__(self, method="none", pseudocount=1e-6):
        self.method = method
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        if self.method not in {"none", "log", "clr"}:
            raise ValueError("method must be one of: 'none', 'log', 'clr'.")
        if not isinstance(self.pseudocount, (int, float)) or self.pseudocount <= 0:
            raise ValueError("pseudocount must be a positive number.")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        if self.method == "none":
            return X

        if self.method == "log":
            return np.log1p(X)

        if self.method == "clr":
            X_pc = X + self.pseudocount
            X_rel = X_pc / X_pc.sum(axis=1, keepdims=True)
            log_x = np.log(X_rel)
            return log_x - log_x.mean(axis=1, keepdims=True)

        raise ValueError("method must be one of: 'none', 'log', 'clr'.")
