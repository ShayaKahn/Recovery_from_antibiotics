# Recovery of the gut microbiota after antibiotic perturbation

## Overview

This repository contains Python code for analyzing recovery patterns of the gut microbiota after antibiotic perturbation. It includes reusable analysis modules, dataset-specific analysis scripts, simulation code, plotting utilities, and tests used to support the accompanying manuscript.

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
| UniFrac analysis | Tests recovery patterns using phylogenetic distance. | `src/host_specific_recovery/statistical_models/unifrac_test.py` |

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
```

Each analysis directory contains dataset-specific scripts, such as `Messaoudene_et_al`, `Palleja_et_al`, `Sewunet_et_al`, and `Yaffe_et_al` where applicable.

Example command:

```sh
python analysis/null_model/Yaffe_et_al/null_model_analysis.py
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
