from pathlib import Path

import numpy as np
from scipy.stats import binom

from src.host_specific_recovery.visualizations.plot_surrogate_data_analysis import plot_binomial_tests


def load_binomial_values(dataset_name: str):
    results_dir = (Path("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/results")
                   / dataset_name / "surrogate_data_analysis")

    ranks = np.load(results_dir / "ranks_mid.npy")
    similarities = np.load(results_dir / "similarities.npy")

    n_subjects = len(ranks)
    n_success = int((ranks == 1).sum())
    prob_null = n_subjects / len(similarities)

    interval_path = results_dir / "binomial_test_interval_null.npy"
    if interval_path.exists():
        interval_null = tuple(np.load(interval_path))
    else:
        interval_null = binom.interval(0.9, n_subjects, prob_null)

    return n_success, n_subjects, prob_null, interval_null


n_success_Sewunet, n_subjects_Sewunet, prob_null_Sewunet, interval_null_Sewunet = load_binomial_values(
    "Sewunet_et_al"
)
n_success_Palleja, n_subjects_Palleja, prob_null_Palleja, interval_null_Palleja = load_binomial_values(
    "Palleja_et_al"
)
n_success_Messaoudene, n_subjects_Messaoudene, prob_null_Messaoudene, interval_null_Messaoudene = load_binomial_values(
    "Messaoudene_et_al"
)

tests = [
    {'k': n_success_Sewunet, 'n': n_subjects_Sewunet, 'p': prob_null_Sewunet, 'interval': interval_null_Sewunet},
    {'k': n_success_Palleja, 'n': n_subjects_Palleja, 'p': prob_null_Palleja, 'interval': interval_null_Palleja},
    {'k': n_success_Messaoudene, 'n': n_subjects_Messaoudene, 'p': prob_null_Messaoudene,
     'interval': interval_null_Messaoudene}
]

plot_binomial_tests(tests, path=None)
