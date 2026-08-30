from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import LeaveOneOut

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.host_specific_recovery.ML_models.linear_regression import MicrobiomeRegularizedLinearRegressor
from src.host_specific_recovery.ML_models.random_forest import MicrobiomeRandomForestRegressor
from src.host_specific_recovery.io.Yaffe_et_al_loader import load_Yaffe_et_al_data
from src.host_specific_recovery.visualizations.plot_SDA_regression_results import plot_prediction_scatter

dataset = load_Yaffe_et_al_data()

subject_ids = dataset["filtered_keys"]
keys = dataset["keys"]
key_to_index = {key: i for i, key in enumerate(keys)}
sample_similarity_indices = np.array([key_to_index[key] for key in subject_ids], dtype=int)

X = dataset["baseline_df"].T.copy()
X.index = subject_ids
X_taxonomic_only = X.add_prefix("taxonomic__")

rank_dir = PROJECT_ROOT / "results" / "Yaffe_et_al" / "surrogate_data_analysis"

similarity_mid = np.load(rank_dir / "similarity_mid.npy")
similarity_others_mid = np.load(rank_dir / "similarity_others_mid.npy")

order = np.load(rank_dir / "order.npy")
standardized_jaccard_sorted = np.load(rank_dir / "obs_sorted.npy")

standardized_jaccard = np.empty_like(standardized_jaccard_sorted, dtype=float)
standardized_jaccard[order] = standardized_jaccard_sorted

cv = LeaveOneOut()

model_configs = [
    #("lasso", {"model_type": "lasso"}),
    ("ridge", {"model_type": "ridge"}),
    # ("elastic_net", {"model_type": "elastic_net", "l1_ratio": 0.5}),
]

search_params_lasso = {
    "linear_model__alpha": np.logspace(-4, 1, 25),
}

search_params_ridge = {
    "linear_model__alpha": np.logspace(-4, 0, 20),
}

search_params_elastic_net = {
    "linear_model__alpha": np.logspace(-4, 0, 20),
}

search_params = [
    search_params_lasso,
    # search_params_ridge,
    # search_params_elastic_net,
]

# prevalence_values = [0.0, 0.1]  # [0.0, 0.1, 0.2, 0.3, 0.4]
analysis_configs = [
    (
        "taxonomic only",
        X_taxonomic_only,
        {
            # "preprocessor__taxonomic__prevalence_filter__min_prevalence": prevalence_values,
        },
    ),
]

feature_importances = {analysis_name: [] for analysis_name, _, _ in analysis_configs}
random_forest_feature_importances = {analysis_name: [] for analysis_name, _, _ in analysis_configs}
random_forest_best_params = {}
validation_performances = {}
prediction_tables = {}

for analysis_name, X_model, branch_search_params in analysis_configs:
    for (model_name, model_kwargs), base_params in zip(model_configs, search_params):
        print(f"\n{model_name}: {analysis_name}")
        params = {**base_params, **branch_search_params}

        regression_model = MicrobiomeRegularizedLinearRegressor(
            min_prevalence=0.3,
            transform="clr",
            alpha=0.1,
            random_state=0,
            n_jobs=8,
            max_iter=10000,
            hyperparameter_search=False,#True,
            search_n_iter=25,
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
            f"Yaffe {model_name} standardized Jaccard: {analysis_name}",
            metrics=performance,
            path=None,
        )

        print(f"{model_name} standardized Jaccard regression ({analysis_name}) validation performance")
        print(performance)

        regression_model.fit(X_model, standardized_jaccard)
        feature_importances[analysis_name].append(np.array(regression_model.feature_importance()))

random_forest_search_params = {
    "random_forest__n_estimators": [200, 500, 1000],
    "random_forest__max_depth": [None, 5, 10, 20],
    "random_forest__min_samples_leaf": [1, 2, 5],
    "random_forest__max_features": ["sqrt", "log2", 0.5],
}

for analysis_name, X_model, _ in analysis_configs:
    model_name = "random_forest"
    print(f"\n{model_name}: {analysis_name}")

    regression_model = MicrobiomeRandomForestRegressor(
        min_prevalence=0.3,
        transform="clr",
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=0,
        n_jobs=8,
        hyperparameter_search=True,
        search_method="randomized",
        search_params=random_forest_search_params,
        search_cv=5,
        search_n_iter=25,
        search_verbose=1,
        avoid_target_leakage=True,
        similarity_mid=similarity_mid,
        similarity_others_mid=similarity_others_mid,
        sample_similarity_indices=sample_similarity_indices,
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
        f"Yaffe random forest standardized Jaccard: {analysis_name}",
        metrics=performance,
        path=None,
    )

    print(f"Random forest standardized Jaccard regression ({analysis_name}) validation performance")
    print(performance)

    regression_model.fit(X_model, standardized_jaccard)
    random_forest_best_params[analysis_name] = regression_model.best_params_
    random_forest_feature_importances[analysis_name].append(regression_model.feature_importance())

print(0)
