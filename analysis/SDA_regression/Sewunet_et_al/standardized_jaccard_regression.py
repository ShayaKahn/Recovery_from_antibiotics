from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import LeaveOneOut

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.host_specific_recovery.ML_models.linear_regression import MicrobiomeRegularizedLinearRegressor
from src.host_specific_recovery.io.Sewunet_et_al_loader import load_Sewunet_et_al_data
from src.host_specific_recovery.visualizations.plot_SDA_regression_results import plot_prediction_scatter

dataset = load_Sewunet_et_al_data()
subject_ids = dataset["filtered_keys"]
keys = dataset["keys"]
key_to_index = {key: i for i, key in enumerate(keys)}
sample_similarity_indices = np.array([key_to_index[key] for key in subject_ids], dtype=int)

X = dataset["abx_df"].T.copy()
X.index = subject_ids
X_taxonomic_only = X.add_prefix("taxonomic__")

rank_dir = PROJECT_ROOT / "results" / "Sewunet_et_al" / "surrogate_data_analysis"

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

search_params_lasso = {
    "linear_model__alpha": np.logspace(-4, 0, 200),
}

search_params = [
    search_params_lasso,
]

prevalence_values = [0.05, 0.1, 0.15]
analysis_configs = [
    ("taxonomic only", X_taxonomic_only,
     {"preprocessor__taxonomic__prevalence_filter__min_prevalence": prevalence_values}),
]

feature_importances = {analysis_name: [] for analysis_name, _, _ in analysis_configs}
validation_performances = {}
prediction_tables = {}

for analysis_name, X_model, branch_search_params in analysis_configs:
    for (model_name, model_kwargs), base_params in zip(model_configs, search_params):
        print(f"\n{model_name}: {analysis_name}")
        params = {**base_params, **branch_search_params}

        regression_model = MicrobiomeRegularizedLinearRegressor(
            min_prevalence=0.1,
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
            f"Sewunet {model_name} standardized Jaccard: {analysis_name}",
            metrics=performance,
            path=rank_dir / f"{model_name}_standardized_jaccard_scatter.png"
        )

        print(f"{model_name} standardized Jaccard regression ({analysis_name}) validation performance")
        print(performance)

        regression_model.fit(X_model, standardized_jaccard)
        feature_importances[analysis_name].append(regression_model.feature_importance())

print(0)
