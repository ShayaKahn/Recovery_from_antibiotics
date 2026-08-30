from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import LeaveOneOut

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.host_specific_recovery.ML_models.linear_regression import MicrobiomeRegularizedLinearRegressor
from src.host_specific_recovery.utils.microbiome_transformers import combine_taxonomic_functional_tables
from src.host_specific_recovery.io.Messaoudene_et_al_loader import (load_Messaoudene_et_al_data,
                                                                    load_Messaoudene_et_al_functional_data)
from src.host_specific_recovery.visualizations.plot_SDA_regression_results import plot_prediction_scatter

dataset = load_Messaoudene_et_al_data()
fun_dataset = load_Messaoudene_et_al_functional_data()
PATH_data = fun_dataset['PATH']
rename_map = fun_dataset['rename_map']

metadata = fun_dataset["metadata"].copy()
metadata = metadata.set_index("Name")

metadata_cols = [
    "host_sex",
    "host_height",
    "host_body_mass_index",
    "Host_age",
    "host_weight",
]

metadata = metadata.loc[~metadata.index.duplicated(keep="first"), metadata_cols]

metadata_X = metadata.loc[dataset["baseline_df"].columns].copy()

PATH_data = PATH_data.rename(columns=rename_map)
PATH_baseline_df = PATH_data[dataset["baseline_df"].columns]
subject_ids = dataset["filtered_keys"]
keys = dataset["keys"]
key_to_index = {key: i for i, key in enumerate(keys)}
sample_similarity_indices = np.array([key_to_index[key] for key in subject_ids], dtype=int)

baseline_df = dataset["baseline_df"]

metadata_X.index = subject_ids

metadata_X["host_sex"] = metadata_X["host_sex"].astype("string").str.lower().map({"male": 0, "female": 1})
continuous_metadata_cols = [column for column in metadata_cols if column != "host_sex"]
metadata_X[continuous_metadata_cols] = metadata_X[continuous_metadata_cols].apply(
    lambda column: np.asarray(column, dtype=float)
)

X = baseline_df.T.copy()
X.index = subject_ids

PATH_X = PATH_baseline_df.T.copy()
PATH_X.index = subject_ids

X_taxonomic_only = X.add_prefix("taxonomic__")
X_functional_only = PATH_X.add_prefix("functional__")
X_taxonomic_functional = combine_taxonomic_functional_tables(X, PATH_X)
X_all = combine_taxonomic_functional_tables(X, PATH_X, metadata_df=metadata_X)

rank_dir = PROJECT_ROOT / "results" / "Messaoudene_et_al" / "surrogate_data_analysis"

similarity_mid = np.load(rank_dir / "similarity_mid.npy")
similarity_others_mid = np.load(rank_dir / "similarity_others_mid.npy")

order = np.load(rank_dir / "order.npy")
standardized_jaccard_sorted = np.load(rank_dir / "obs_sorted.npy")

standardized_jaccard = np.empty_like(standardized_jaccard_sorted, dtype=float)
standardized_jaccard[order] = standardized_jaccard_sorted

cv = LeaveOneOut()

model_configs = [
    ("lasso", {"model_type": "lasso"}),
]

search_params_lasso = {"linear_model__alpha": np.logspace(-4, 0, 200)}

search_params = [
    search_params_lasso,
]

prevalence_values = [0.2, 0.25, 0.3, 0.35]
analysis_configs = [
    ("taxonomic only", X_taxonomic_only,
     {"preprocessor__taxonomic__prevalence_filter__min_prevalence": prevalence_values}),
    #("functional PATH only", X_functional_only,
    # {"preprocessor__functional__prevalence_filter__min_prevalence": prevalence_values}),
    #("taxonomic and functional PATH", X_taxonomic_functional,
    # {"preprocessor__taxonomic__prevalence_filter__min_prevalence": prevalence_values,
    # "preprocessor__functional__prevalence_filter__min_prevalence": prevalence_values}),
]

feature_importances = {analysis_name: [] for analysis_name, _, _ in analysis_configs}
validation_performances = {}
prediction_tables = {}

for analysis_name, X_model, branch_search_params in analysis_configs:
    for (model_name, model_kwargs), base_params in zip(model_configs, search_params):
        print(f"\n{model_name}: {analysis_name}")
        params = {**base_params, **branch_search_params}

        regression_model = MicrobiomeRegularizedLinearRegressor(
            min_prevalence=0.3,
            functional_min_prevalence=0.,
            functional_transform="clr",
            transform="clr",
            alpha=0.01,
            random_state=0,
            n_jobs=8,
            max_iter=10000,
            hyperparameter_search=True,
            search_n_iter=200,
            avoid_target_leakage=True,
            similarity_mid=similarity_mid,
            similarity_others_mid=similarity_others_mid,
            sample_similarity_indices=sample_similarity_indices,
            search_params=params,
            search_verbose=1,
            search_method="grid",
            **model_kwargs,
        )

        performance, predictions = regression_model.evaluate_cv_predictions(
            X_model,
            None,
            cv=cv,
            return_predictions=True,
        )

        result_key = (model_name, analysis_name)
        validation_performances[result_key] = performance
        prediction_tables[result_key] = predictions

        plot_prediction_scatter(
            predictions,
            f"Messaoudene {model_name} standardized Jaccard: {analysis_name}",
            metrics=performance,
            path=rank_dir / f"{model_name}_{analysis_name}_standardized_jaccard_scatter.png",
        )

        print(f"{model_name} standardized Jaccard regression ({analysis_name}) validation performance")
        print(performance)

        regression_model.fit(X_model, standardized_jaccard)
        feature_importances[analysis_name].append(regression_model.feature_importance())

print(0)
