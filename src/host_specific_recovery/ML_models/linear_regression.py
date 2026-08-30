import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV, RepeatedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import spearmanr

from src.host_specific_recovery.utils.microbiome_transformers import (
    FUNCTIONAL_PREFIX,
    METADATA_PREFIX,
    TAXONOMIC_PREFIX,
    AbundanceTransformer,
    PrevalenceFilter,
    _select_binary_numeric_columns,
    _select_continuous_numeric_columns,
)


class MicrobiomeRegularizedLinearRegressor:
    """Regularized linear regressor for microbiome abundance tables.

    Microbial branches receive prevalence filtering, an optional abundance
    transformation, and standardization. Continuous metadata columns receive
    median imputation and standardization. Numeric binary metadata columns
    receive most-frequent imputation without scaling, while categorical columns
    receive most-frequent imputation and one-hot encoding.

    Parameters:
    avoid_target_leakage : bool, default=False
        If True, evaluate_cv_predictions recomputes the standardized SDA target
        inside each external CV split instead of using the precomputed y values.
        For every training sample, the target is calculated from similarity_mid
        and similarity_others_mid after excluding the held-out test samples from
        the surrogate comparison vector.
    similarity_mid : array-like of shape (n_samples,), optional
        Observed SDA similarity for each sample. This is the output saved by
        save_sda_results as "similarity_mid.npy".
    similarity_others_mid : array-like of shape (n_samples, n_samples - 1), optional
        Surrogate/other-baseline SDA similarities for each model sample. Rows
        must be aligned with X/y. Columns are indexed by the full dataset keys,
        with the row subject omitted.
    sample_similarity_indices : array-like of shape (n_model_samples,), optional
        Maps each row in X/y to the matching subject index in the full keys
        list used to build the columns of similarity_others_mid. Use this when
        similarity_others_mid includes additional independent baseline samples
        that are not rows in X/y.
    The default model_type is "lasso" for backward compatibility."""

    def __init__(self, min_prevalence=0.1, transform="none", pseudocount=1e-6,
                 functional_min_prevalence=None, functional_transform=None,
                 model_type="lasso", alpha=1.0, l1_ratio=0.5,
                 fit_intercept=True, max_iter=10000, tol=1e-4, selection="cyclic",
                 random_state=0, n_jobs=-1, hyperparameter_search=False, search_method="randomized",
                 search_params=None, search_scoring="neg_mean_squared_error", search_cv=5,
                 search_n_iter=50, search_refit=True, search_verbose=0,
                 avoid_target_leakage=False, similarity_mid=None, similarity_others_mid=None,
                 sample_similarity_indices=None):

        self.min_prevalence = min_prevalence
        self.transform = transform
        self.pseudocount = pseudocount
        self.functional_min_prevalence = functional_min_prevalence
        self.functional_transform = functional_transform
        self.model_type = self._normalize_model_type(model_type)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.hyperparameter_search = hyperparameter_search
        self.search_method = search_method
        self.search_params = search_params
        self.search_scoring = search_scoring
        self.search_cv = search_cv
        self.search_n_iter = search_n_iter
        self.search_refit = search_refit
        self.search_verbose = search_verbose
        self.avoid_target_leakage = avoid_target_leakage
        self.similarity_mid = similarity_mid
        self.similarity_others_mid = similarity_others_mid
        self.sample_similarity_indices = sample_similarity_indices
        self._validate_parameters()

    @staticmethod
    def _normalize_model_type(model_type):
        if model_type == "elasticnet":
            return "elastic_net"
        return model_type

    def _validate_parameters(self):
        if not isinstance(self.min_prevalence, (int, float)) or not (0 <= self.min_prevalence <= 1):
            raise ValueError("min_prevalence must be a number between 0 and 1.")
        if self.functional_min_prevalence is not None and (
                not isinstance(self.functional_min_prevalence, (int, float))
                or not (0 <= self.functional_min_prevalence <= 1)
        ):
            raise ValueError("functional_min_prevalence must be None or a number between 0 and 1.")
        if self.transform not in {"none", "log", "clr"}:
            raise ValueError("transform must be one of: 'none', 'log', 'clr'.")
        if self.functional_transform is not None and self.functional_transform not in {"none", "log", "clr"}:
            raise ValueError("functional_transform must be None or one of: 'none', 'log', 'clr'.")
        if not isinstance(self.pseudocount, (int, float)) or self.pseudocount <= 0:
            raise ValueError("pseudocount must be a positive number.")
        if self.model_type not in {"lasso", "ridge", "elastic_net"}:
            raise ValueError("model_type must be one of: 'lasso', 'ridge', 'elastic_net'.")
        if not isinstance(self.alpha, (int, float)) or self.alpha < 0:
            raise ValueError("alpha must be a non-negative number.")
        if not isinstance(self.l1_ratio, (int, float)) or not (0 <= self.l1_ratio <= 1):
            raise ValueError("l1_ratio must be a number between 0 and 1.")
        if not isinstance(self.fit_intercept, bool):
            raise TypeError("fit_intercept must be bool.")
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(self.tol, (int, float)) or self.tol <= 0:
            raise ValueError("tol must be a positive number.")
        if self.selection not in {"cyclic", "random"}:
            raise ValueError("selection must be one of: 'cyclic', 'random'.")
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer or None.")
        if self.n_jobs is not None and not isinstance(self.n_jobs, int):
            raise TypeError("n_jobs must be an integer or None.")
        if not isinstance(self.hyperparameter_search, bool):
            raise TypeError("hyperparameter_search must be bool.")
        if self.search_method not in {"grid", "randomized"}:
            raise ValueError("search_method must be one of: 'grid', 'randomized'.")
        if self.search_params is not None and not isinstance(self.search_params, dict):
            raise TypeError("search_params must be a dict or None.")
        if not (isinstance(self.search_scoring, str) or callable(self.search_scoring)):
            raise TypeError("search_scoring must be a string or callable.")
        if not isinstance(self.search_cv, int) or self.search_cv < 2:
            raise ValueError("search_cv must be an integer >= 2.")
        if not isinstance(self.search_n_iter, int) or self.search_n_iter <= 0:
            raise ValueError("search_n_iter must be a positive integer.")
        if not isinstance(self.search_refit, bool):
            raise TypeError("search_refit must be bool.")
        if not isinstance(self.search_verbose, int) or self.search_verbose < 0:
            raise ValueError("search_verbose must be a non-negative integer.")
        if not isinstance(self.avoid_target_leakage, bool):
            raise TypeError("avoid_target_leakage must be bool.")

    def _build_linear_model(self):
        common_params = {"alpha": self.alpha, "fit_intercept": self.fit_intercept, "max_iter": self.max_iter,
                         "tol": self.tol}

        if self.model_type == "lasso":
            return Lasso(**common_params, selection=self.selection, random_state=self.random_state)

        if self.model_type == "ridge":
            return Ridge(**common_params, random_state=self.random_state)

        if self.model_type == "elastic_net":
            return ElasticNet(**common_params, l1_ratio=self.l1_ratio, selection=self.selection,
                              random_state=self.random_state)

        raise ValueError("model_type must be one of: 'lasso', 'ridge', 'elastic_net'.")

    @staticmethod
    def _build_branch(min_prevalence, transform, pseudocount):
        return Pipeline(
            steps=[
                ("prevalence_filter", PrevalenceFilter(min_prevalence)),
                ("abundance_transform", AbundanceTransformer(transform, pseudocount)),
                ("standard_scaler", StandardScaler()),
            ]
        )

    @staticmethod
    def _build_metadata_branch():
        continuous_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("standard_scaler", StandardScaler()),
            ]
        )
        binary_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("continuous", continuous_pipeline, _select_continuous_numeric_columns),
                ("binary", binary_pipeline, _select_binary_numeric_columns),
                ("categorical", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
            ],
            sparse_threshold=0,
        )

    @staticmethod
    def _split_column_types(X):
        if not hasattr(X, "columns"):
            raise TypeError(
                "X must be a pandas.DataFrame with 'taxonomic__', 'functional__', and/or 'metadata__' "
                "prefixed columns, e.g. as produced by combine_taxonomic_functional_tables."
            )
        taxonomic_columns = [column for column in X.columns if column.startswith(TAXONOMIC_PREFIX)]
        functional_columns = [column for column in X.columns if column.startswith(FUNCTIONAL_PREFIX)]
        metadata_columns = [column for column in X.columns if column.startswith(METADATA_PREFIX)]
        if not taxonomic_columns and not functional_columns and not metadata_columns:
            raise ValueError(
                f"X must contain at least one column with the '{TAXONOMIC_PREFIX}' or "
                f"'{FUNCTIONAL_PREFIX}' or '{METADATA_PREFIX}' prefix."
            )
        return taxonomic_columns, functional_columns, metadata_columns

    def _build_pipeline(self, X):
        taxonomic_columns, functional_columns, metadata_columns = self._split_column_types(X)

        transformers = []

        if taxonomic_columns:
            transformers.append(
                ("taxonomic", self._build_branch(self.min_prevalence, self.transform, self.pseudocount),
                 taxonomic_columns)
            )

        if functional_columns:
            functional_min_prevalence = (
                self.min_prevalence
                if self.functional_min_prevalence is None
                else self.functional_min_prevalence
            )
            functional_transform = self.transform if self.functional_transform is None else self.functional_transform
            transformers.append(
                ("functional", self._build_branch(functional_min_prevalence, functional_transform, self.pseudocount),
                functional_columns)
            )

        if metadata_columns:
            transformers.append(
                ("metadata", self._build_metadata_branch(), metadata_columns)
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        return Pipeline(steps=[("preprocessor", preprocessor), ("linear_model", self._build_linear_model())])

    def _default_search_params(self):
        params = {"linear_model__alpha": np.logspace(-4, 0, 30),
                  "linear_model__fit_intercept": [True, False]}

        if self.model_type in {"lasso", "elastic_net"}:
            params["linear_model__selection"] = ["cyclic", "random"]

        if self.model_type == "elastic_net":
            params["linear_model__l1_ratio"] = [0.1, 0.25, 0.5, 0.75, 0.9]

        return params

    def _build_search_estimator(self, X, y):
        pipeline = self._build_pipeline(X)

        if not self.hyperparameter_search:
            return pipeline

        if not self.search_refit:
            raise ValueError("search_refit must be True so the best pipeline can be used after fitting.")

        search_cv = min(self.search_cv, y.shape[0])
        if search_cv < 2:
            raise ValueError("At least 2 samples are required for hyperparameter search.")

        cv = KFold(n_splits=search_cv, shuffle=True, random_state=self.random_state)
        search_params = self.search_params or self._default_search_params()

        if self.search_method == "grid":
            return GridSearchCV(estimator=pipeline, param_grid=search_params, scoring=self.search_scoring, cv=cv,
                                n_jobs=self.n_jobs, refit=self.search_refit, verbose=self.search_verbose,
                                error_score="raise")

        if self.search_method == "randomized":
            return RandomizedSearchCV(estimator=pipeline, param_distributions=search_params, n_iter=self.search_n_iter,
                                      scoring=self.search_scoring, cv=cv, n_jobs=self.n_jobs, refit=self.search_refit,
                                      random_state=self.random_state, verbose=self.search_verbose, error_score="raise")

        raise ValueError("search_method must be one of: 'grid', 'randomized'.")

    @staticmethod
    def _validate_X(X):
        if not isinstance(X, pd.DataFrame):
            return PrevalenceFilter._to_numpy(X)

        taxonomic_columns, functional_columns, _ = (
            MicrobiomeRegularizedLinearRegressor._split_column_types(X)
        )
        microbial_columns = taxonomic_columns + functional_columns
        if microbial_columns:
            PrevalenceFilter._to_numpy(X[microbial_columns])

        return X.to_numpy(dtype=object)

    @staticmethod
    def _validate_y(y):
        if not isinstance(y, (pd.Series, pd.Index, list, tuple, np.ndarray)):
            raise TypeError("y must be a pandas.Series, pandas.Index, list, tuple, or 1D numpy.ndarray.")

        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("y must be a 1D vector of numeric regression targets.")
        if y.size == 0:
            raise ValueError("y must contain at least one target value.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite, non-missing numeric values.")

        return y

    def _validate_similarity_targets(self, n_model_samples):
        if self.similarity_mid is None or self.similarity_others_mid is None:
            raise ValueError(
                "similarity_mid and similarity_others_mid must be provided when avoid_target_leakage=True."
            )

        similarity_mid = np.asarray(self.similarity_mid, dtype=float)
        similarity_others_mid = np.asarray(self.similarity_others_mid, dtype=float)

        if similarity_mid.ndim != 1:
            raise ValueError("similarity_mid must be a 1D array.")
        if similarity_mid.shape[0] != n_model_samples:
            raise ValueError("similarity_mid must contain one value per row in X.")
        if similarity_others_mid.ndim != 2 or similarity_others_mid.shape[0] != n_model_samples:
            raise ValueError(
                "similarity_others_mid must be a 2D array with one row per row in X."
            )
        if not np.all(np.isfinite(similarity_mid)) or not np.all(np.isfinite(similarity_others_mid)):
            raise ValueError("similarity_mid and similarity_others_mid must contain only finite values.")

        n_similarity_samples = similarity_others_mid.shape[1] + 1

        if self.sample_similarity_indices is None:
            sample_similarity_indices = np.arange(n_model_samples, dtype=int)
        else:
            sample_similarity_indices = np.asarray(self.sample_similarity_indices, dtype=int)

        if sample_similarity_indices.ndim != 1 or sample_similarity_indices.shape[0] != n_model_samples:
            raise ValueError("sample_similarity_indices must be a 1D array with one index per row in X.")
        if np.unique(sample_similarity_indices).shape[0] != sample_similarity_indices.shape[0]:
            raise ValueError("sample_similarity_indices must not contain duplicate indexes.")
        if sample_similarity_indices.min() < 0 or sample_similarity_indices.max() >= n_similarity_samples:
            raise ValueError(
                "sample_similarity_indices must refer to valid subject indexes in the full similarity_others_mid "
                "column universe."
            )

        return similarity_mid, similarity_others_mid, sample_similarity_indices

    @staticmethod
    def _similarity_other_columns(row_index, allowed_indices):
        return [idx if idx < row_index else idx - 1 for idx in allowed_indices if idx != row_index]

    def _calculate_standardized_similarity_target(self, row_index, row_similarity_index, allowed_indices,
                                                 similarity_mid, similarity_others_mid):
        other_columns = self._similarity_other_columns(row_similarity_index, allowed_indices)
        if len(other_columns) == 0:
            raise ValueError("At least one comparison sample is required to calculate the SDA target.")

        other_similarities = similarity_others_mid[row_index, other_columns]
        sd = other_similarities.std(ddof=0)
        if sd == 0:
            raise ValueError("Cannot calculate standardized SDA target because comparison std is zero.")

        return (similarity_mid[row_index] - other_similarities.mean()) / sd

    def _calculate_fold_targets(self, train_indices, test_indices, similarity_mid, similarity_others_mid,
                                sample_similarity_indices):
        train_similarity_indices = sample_similarity_indices[train_indices]
        test_similarity_indices = sample_similarity_indices[test_indices]
        allowed_indices = np.setdiff1d(
            np.arange(similarity_others_mid.shape[1] + 1, dtype=int),
            test_similarity_indices,
            assume_unique=False,
        )

        y_train = np.asarray([
            self._calculate_standardized_similarity_target(
                row_index, row_similarity_index, allowed_indices, similarity_mid, similarity_others_mid
            )
            for row_index, row_similarity_index in zip(train_indices, train_similarity_indices)
        ], dtype=float)

        y_test = np.asarray([
            self._calculate_standardized_similarity_target(
                row_index, row_similarity_index, allowed_indices, similarity_mid, similarity_others_mid
            )
            for row_index, row_similarity_index in zip(test_indices, test_similarity_indices)
        ], dtype=float)

        return y_train, y_test

    @staticmethod
    def _slice_samples(X, indices):
        if hasattr(X, "iloc"):
            return X.iloc[indices]
        return np.asarray(X)[indices]

    def fit(self, X, y):
        X_arr = self._validate_X(X)
        y = self._validate_y(y)

        if X_arr.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        estimator = self._build_search_estimator(X, y)
        estimator.fit(X, y)

        if self.hyperparameter_search:
            self.search_ = estimator
            self.best_params_ = estimator.best_params_
            self.best_score_ = estimator.best_score_
            self.cv_results_ = estimator.cv_results_
            self.pipeline_ = estimator.best_estimator_
        else:
            self.pipeline_ = estimator

        return self

    def predict(self, X):
        self._check_is_fitted()
        self._validate_X(X)
        return self.pipeline_.predict(X)

    @staticmethod
    def _pearson_correlation(y_true, y_pred):
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return np.nan
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    @staticmethod
    def _spearman_correlation(y_true, y_pred):
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return np.nan
        return float(spearmanr(y_true, y_pred).correlation)

    def evaluate_cv_predictions(self, X, y, cv, return_predictions=False):
        X_arr = self._validate_X(X)
        if self.avoid_target_leakage and y is None:
            y = np.zeros(X_arr.shape[0], dtype=float)
        else:
            y = self._validate_y(y)

        if X_arr.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if y.shape[0] < 2:
            raise ValueError("At least 2 samples are required for cross-validation.")

        if self.avoid_target_leakage:
            similarity_mid, similarity_others_mid, sample_similarity_indices = self._validate_similarity_targets(
                X_arr.shape[0]
            )
            y_true = np.empty(X_arr.shape[0], dtype=float)
            y_pred = np.empty(X_arr.shape[0], dtype=float)

            for train_indices, test_indices in cv.split(X_arr, y):
                train_indices = np.asarray(train_indices, dtype=int)
                test_indices = np.asarray(test_indices, dtype=int)
                y_train, y_test = self._calculate_fold_targets(
                    train_indices, test_indices, similarity_mid, similarity_others_mid,
                    sample_similarity_indices
                )

                X_train = self._slice_samples(X, train_indices)
                X_test = self._slice_samples(X, test_indices)

                estimator = self._build_search_estimator(X_train, y_train)
                estimator.fit(X_train, y_train)

                y_true[test_indices] = y_test
                y_pred[test_indices] = estimator.predict(X_test)
        else:
            y_true = y
            y_pred = cross_val_predict(self._build_search_estimator(X, y), X, y, cv=cv, n_jobs=self.n_jobs)

        metrics = [
            {"metric": "r2", "value": r2_score(y_true, y_pred), "n_samples": y_true.shape[0]},
            {"metric": "mean_absolute_error", "value": mean_absolute_error(y_true, y_pred),
             "n_samples": y_true.shape[0]},
            {"metric": "mean_squared_error", "value": mean_squared_error(y_true, y_pred),
             "n_samples": y_true.shape[0]},
            {"metric": "pearson_correlation", "value": self._pearson_correlation(y_true, y_pred),
             "n_samples": y_true.shape[0]},
            {"metric": "spearman_correlation", "value": self._spearman_correlation(y_true, y_pred),
             "n_samples": y_true.shape[0]}
        ]

        metrics_df = pd.DataFrame(metrics)

        if return_predictions:
            predictions_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            return metrics_df, predictions_df

        return metrics_df

    def evaluate_cv(self, X, y, n_splits=5, n_repeats=20):
        X_arr = self._validate_X(X)
        y = self._validate_y(y)

        if X_arr.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2.")
        if not isinstance(n_repeats, int) or n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer.")
        if y.shape[0] < 2:
            raise ValueError("At least 2 samples are required for cross-validation.")

        n_splits = min(n_splits, y.shape[0])

        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)

        scoring = {
            "r2": "r2",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
            "neg_mean_squared_error": "neg_mean_squared_error",
            "pearson_correlation": make_scorer(self._pearson_correlation),
            "spearman_correlation": make_scorer(self._spearman_correlation),
        }

        results = cross_validate(self._build_search_estimator(X, y), X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs,
                                 error_score="raise")

        summary = []
        for key, values in results.items():
            if key.startswith("test_"):
                metric = key.replace("test_", "")
                values = np.asarray(values, dtype=float)

                if metric == "neg_mean_absolute_error":
                    metric = "mean_absolute_error"
                    values = -values
                elif metric == "neg_mean_squared_error":
                    metric = "mean_squared_error"
                    values = -values

                summary.append(
                    {"metric": metric, "mean": np.mean(values), "std": np.std(values)})

        return pd.DataFrame(summary)

    def feature_importance(self, top_n=None):
        self._check_is_fitted()

        linear_model = self.pipeline_.named_steps["linear_model"]
        preprocessor = self.pipeline_.named_steps["preprocessor"]

        feature_types = []
        feature_names = []
        for branch_name, prefix in (("taxonomic", TAXONOMIC_PREFIX), ("functional", FUNCTIONAL_PREFIX)):
            if branch_name not in preprocessor.named_transformers_:
                continue
            prevalence_filter = preprocessor.named_transformers_[branch_name].named_steps["prevalence_filter"]
            branch_names = prevalence_filter.get_feature_names_out()
            feature_names.extend(
                name[len(prefix):] if name.startswith(prefix) else name for name in branch_names
            )
            feature_types.extend([branch_name] * len(branch_names))

        if "metadata" in preprocessor.named_transformers_:
            metadata_transformer = preprocessor.named_transformers_["metadata"]
            metadata_names = metadata_transformer.get_feature_names_out()
            for name in metadata_names:
                # Nested ColumnTransformer names look like
                # "numeric__metadata__age" or "categorical__metadata__sex_female".
                name = name.split("__", 1)[-1]
                if name.startswith(METADATA_PREFIX):
                    name = name[len(METADATA_PREFIX):]
                feature_names.append(name)
                feature_types.append("metadata")

        coefficients = linear_model.coef_
        df = pd.DataFrame(
            {
                "feature_type": feature_types,
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        ).sort_values("abs_coefficient", ascending=False)

        if top_n is not None:
            df = df.head(top_n)

        return df.reset_index(drop=True)

    def _check_is_fitted(self):
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("Model is not fitted yet. Run .fit(X, y) first.")

MicrobiomeLassoRegressor = MicrobiomeRegularizedLinearRegressor
MicrobiomeRegularizedRegressor = MicrobiomeRegularizedLinearRegressor
