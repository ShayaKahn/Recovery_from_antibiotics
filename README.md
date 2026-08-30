# Recovery of the gut microbiota after antibiotic perturbation

## Overview

This repository contains Python code for analyzing recovery patterns of the gut microbiota after antibiotic perturbation. It includes reusable analysis and machine-learning modules, dataset-specific analysis scripts, simulation code, plotting utilities, and tests used to support the accompanying manuscript.

The project focuses on questions such as whether microbiota recovery is personalized, whether recovery can be explained by random sampling from a baseline cohort, and whether surviving species after antibiotic treatment influence later community assembly.

## Manuscript

The current version of the research article is available here:

[Manuscript PDF](https://drive.google.com/file/d/1yvUP08y3Mzgn8f729OFHlFhsNsIvKxsl/view?usp=sharing)

## Installation

Clone the repository and create a virtual environment:

```sh
git clone https://github.com/ShayaKahn/Recovery_from_antibiotics
cd Recovery_from_antibiotics
python -m venv .venv
```

Activate the environment:

```sh
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project was developed with Python 3.12. Other recent Python 3 versions may work, but Python 3.12 is the recommended environment.

## Quick Start

Most reusable functionality is implemented under `src/host_specific_recovery/`. For example, the null model can be imported and applied directly:

```python
import numpy as np

from src.host_specific_recovery.statistical_models.null_model import NullModel

baseline_sample = np.array([1, 0, 1, 1, 0])
abx_sample = np.array([0, 0, 1, 0, 0])
baseline_cohort = np.array([
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 1, 1],
])
post_abx_matrix = np.array([
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
])

model = NullModel(
    baseline_sample,
    abx_sample,
    baseline_cohort,
    post_abx_matrix,
    num_reals=100,
    timepoints=1,
)

real_similarity, synthetic_similarities = 1 - model.distance(method="Jaccard")
```

Dataset-specific scripts are located under `analysis/`. These scripts are intended for reproducing manuscript analyses when the corresponding input data files are available.

## Repository Structure

```text
src/host_specific_recovery/   Core Python package
src/host_specific_recovery/ML_models/   Microbiome regression models
analysis/                     Dataset-specific analysis scripts
scripts/                      External workflow and helper scripts
tests/                        Unit tests
results/                      Generated outputs, when present
utils/                        General helper functions
cython_modules/               Cython extensions and compiled modules
```

## Main Methods

| Method | Purpose | Main implementation |
| --- | --- | --- |
| Null model | Tests whether recovery can be explained by random sampling from a baseline distribution. | `src/host_specific_recovery/statistical_models/null_model.py` |
| Surrogate data analysis | Compares a subject's post-antibiotic state to their own baseline and to other subjects' baselines. | `src/host_specific_recovery/statistical_models/surrogate.py` |
| Similarity correlation | Tests relationships between surviving species and newly appearing species after treatment. | `src/host_specific_recovery/statistical_models/similarity_correlation.py` |
| Historical contingency simulation | Simulates how post-antibiotic community assembly can depend on species that survived treatment. | `src/host_specific_recovery/simulations/historical_contingency.py` |
| Functional analysis | Analyzes functional recovery patterns using predicted metagenomic contribution data. | `src/host_specific_recovery/statistical_models/functional_test.py` |
| Regularized linear regression | Primary machine-learning model for predicting continuous recovery outcomes from taxonomic, functional, and metadata features. Supports Lasso, Ridge, and Elastic Net regression with microbiome-aware preprocessing, cross-validation, optional hyperparameter search, and coefficient interpretation. | `src/host_specific_recovery/ML_models/linear_regression.py` |

## Primary Machine-Learning Model: Regularized Linear Regression

`MicrobiomeRegularizedLinearRegressor` is the project's main machine-learning model. It accepts a pandas DataFrame whose columns use one or more of the following prefixes:

- `taxonomic__` for taxonomic abundance features
- `functional__` for functional abundance features
- `metadata__` for continuous, binary, or categorical subject metadata

Microbial features can be prevalence-filtered and transformed using `none`, `log`, or `clr`. Metadata are imputed and encoded according to their data type. The estimator supports Lasso, Ridge, and Elastic Net regularization, as well as grid or randomized hyperparameter search, repeated cross-validation, out-of-fold predictions, and ranked model coefficients.

```python
from src.host_specific_recovery.ML_models.linear_regression import (
    MicrobiomeRegularizedLinearRegressor,
)

X = baseline_abundance_table.T.add_prefix("taxonomic__")
y = recovery_scores

model = MicrobiomeRegularizedLinearRegressor(
    min_prevalence=0.3,
    transform="clr",
    model_type="ridge",
    alpha=0.1,
    random_state=0,
)

model.fit(X, y)
predictions = model.predict(X)
cv_summary = model.evaluate_cv(X, y, n_splits=5, n_repeats=20)
important_features = model.feature_importance(top_n=20)
```

For standardized surrogate data analysis (SDA) targets, set `avoid_target_leakage=True` and provide `similarity_mid`, `similarity_others_mid`, and, when necessary, `sample_similarity_indices`. In this mode, `evaluate_cv_predictions` recalculates the target inside each external cross-validation split so held-out subjects are excluded from the surrogate comparison distribution.

## Analysis Scripts

The `analysis/` directory contains scripts organized by analysis type and dataset. Examples include:

```text
analysis/null_model/
analysis/surrogate/
analysis/similarity/
analysis/similarity_correlation/
analysis/survived_species_analysis/
analysis/assembly_times_analysis/
analysis/historical_contingency_simulation/
analysis/SDA_regression/
```

Each analysis directory contains dataset-specific scripts, such as `Messaoudene_et_al`, `Palleja_et_al`, `Sewunet_et_al`, and `Yaffe_et_al` where applicable.

Example command:

```sh
python analysis/null_model/Yaffe_et_al/null_model_analysis.py
```

The Yaffe et al. standardized-Jaccard regression workflow using the regularized linear model is located at:

```text
analysis/SDA_regression/Yaffe_et_al/standardized_jaccard_regression.py
```

Some scripts expect local input data files and output directories. See the data availability section before running manuscript-level analyses.

## Data Availability

Input data files are not included directly in this repository. They can be delivered upon request.

The analysis scripts assume that the required microbiome tables, metadata files, and derived functional or phylogenetic inputs are available locally. Because some scripts are dataset-specific, file paths may need to be adjusted to match the local data location before reproducing the full manuscript analyses.

## Reproducing Results

To reproduce manuscript analyses:

1. Install the Python environment using `requirements.txt`.
2. Obtain the required input data files by request.
3. Place the data files in the expected local structure or update the paths in the relevant dataset loader/script.
4. Run the relevant scripts under `analysis/`.
5. Use the plotting utilities under `src/host_specific_recovery/visualizations/` to generate figures from the produced results.

The dataset loaders are located in:

```text
src/host_specific_recovery/io/
```

These modules define how raw or processed dataset files are loaded into the analysis pipeline.

## Testing

Run the test suite with:

```sh
python -m unittest discover -s tests -p "test*.py"
```

The tests cover core metrics, data-processing utilities, statistical models, and simulations.

## Citation

If you use this repository, please cite the accompanying manuscript. A formal citation will be added when available.

## License

No license file is currently included. Add a license before distributing or reusing this code publicly.
