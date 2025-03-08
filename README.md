# Recovery of the gut microbiota after antibiotic perturbation

## Overview

This project contains a code library for the analysis of the recovery of the microbiota after antibiotic treatment using real and simulated data.
The project implements the null model, surrogate data analysis, similarity correlation analysis, and
historical contingency analysis. This code library is designed to derive insights into the recovery pattern of the gut
microbiota after antibiotic perturbation and the mechanisms related to this pattern using various
statistical techniques.

## Manuscript
The current version of the research article, available here:
[**Manuscript PDF**](https://drive.google.com/file/d/1Os9LfP8WslkpTtsGlqyrRcM4_5EdoVA9/view?usp=sharing).

## Methods
In this repository, I provide implementations of the following methods:

### Null model
The null model tests the following null hypothesis:

**H<sub>0</sub>**: The recovery pattern after antibiotic treatment is random sampling from the baseline distribution.

To test the null hypothesis for a given test subject we generate k random post-antibiotic treatment steady states where
the species at each steady state are randomly sampled from a baseline distribution according to the species presence
frequency in the baseline cohort (baseline samples of different subjects). By measuring k similarity values between the
baseline state of the test subject and the k random steady states, one can test the null hypothesis using a statistical
test by comparing the k similarity values to the real similarity value between the baseline state and the
post-antibiotic treatment state of the test subject.

The null model is implemented in the `null_model.py` script.

#### Usage Example

```python
from methods.null_model import NullModel
import numpy as np

# Define the input data
baseline_sample = np.array([0.1, 0, 0, 0.2, 0.3, 0.1, 0, 0.1, 0.2, 0])
ABX_sample = np.array([0, 0, 0.3, 0.1, 0, 0.1, 0, 0, 0, 0.5])
test_matrix = np.array([[0.5, 0, 0.1, 0, 0.2, 0.2, 0, 0, 0, 0],
                        [0.5, 0, 0.1, 0, 0.2, 0.1, 0, 0.1, 0, 0]])
baseline = np.array([[0.1, 0, 0, 0.2, 0.3, 0.1, 0, 0.1, 0.2, 0],
                     [0.5, 0.1, 0, 0, 0.1, 0.1, 0, 0.1, 0.1, 0],
                     [0.1, 0, 0.2, 0.1, 0, 0.1, 0, 0, 0.2, 0.3]])

# Define the parameters
num_reals = 3
timepoints = 1

# Calculate the similarity values
NM = NullModel(baseline_sample, ABX_sample, baseline,
               test_matrix, num_reals, timepoints)
S_real, S_syn = 1 - NM.distance(method="Specificity")
```

### Surrogate data analysis
The surrogate data analysis measures the similarity between the observed
post-antibiotics state of a test subject and the baseline state of other subjects
and the test subjects' baseline state. By comparing the similarity values to different subjects,
one can measure if the test subject shows a 'Personalized recovery' pattern.
The surrogate data analysis is implemented in the `surrogate.py` script.

#### Usage Example

```python
from methods.surrogate import Surrogate
import numpy as np

# Define the input data
test_post_abx_matrix = np.array([[1, 0, 1, 0, 0, 0, 0, 1],
                                 [1, 0, 1, 1, 0, 1, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 1, 1],
                                 [1, 1, 1, 1, 1, 1, 1, 0]])
test_base_samples_collection = {"Test subject": np.array([[1, 0, 1, 1, 1, 0, 1, 1]])}
test_base_samples_collection_mat = {"Test subject": np.array([[1, 0, 1, 1, 1, 0, 1, 1],
                                                              [1, 0, 1, 0, 1, 0, 1, 1],
                                                              [1, 0, 1, 1, 1, 1, 0, 1]])}
base_samples_collections = {"Subject A": np.array([[0, 1, 1, 1, 0, 1, 1, 0]]),
                            "Subject B": np.array([[1, 1, 0, 1, 1, 0, 0, 1]])}
base_samples_collections_mat = {"Subject A": np.array([[0, 1, 1, 1, 0, 1, 1, 0],
                                                       [0, 1, 1, 1, 0, 1, 0, 0]]),
                                "Subject B": np.array([[1, 1, 0, 1, 1, 0, 0, 1],
                                                       [1, 1, 0, 1, 1, 0, 0, 0]])}
test_abx_sample = np.array([[1, 1, 0, 0, 0, 0, 0, 1]])

# Define the parameters
timepoints = 1

# Calculate the results
SU = Surrogate(base_samples_collections, test_base_samples_collection, test_post_abx_matrix, 
               test_abx_sample, timepoints=timepoints)
results = SU.apply_surrogate_data_analysis()
```

### Similarity correlation
The similarity correlation analysis measures if there is a relation between new species
that appear after antibiotics treatment and the species that are considered as they survived the antibiotic treatment.
This analysis checks for evidence of the 'Historical contingency' mechanism.
The analysis is implemented in the `similarity_correlation.py` script.

#### Usage Example

```python
from methods.similarity_correlation import SimilarityCorrelation
import numpy as np
import pandas as pd

keys = ["A", "B", "C", "D", "E"]
abx = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
base = np.array([[1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                 [1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                 [0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                 [1, 1, 1, 0, 1, 0, 0, 1, 1, 1]])
post_ABX_container = {"A": np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                                     [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                                     [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]]),
                     "B": np.array([[1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1, 1, 0, 0, 1, 1]]),
                     "C": np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                                    [0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                                    [0, 1, 0, 1, 1, 1, 0, 1, 1, 1]]),
                     "D": np.array([[1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                                    [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                                    [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]]),
                     "E": np.array([[1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                                    [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
                                    [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]])
                     }
keys_ref = ["A", "B", "C", "D", "E", "F", "G"]
base_others = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
base_ref = np.vstack([base, base_others])
baseline_ref = pd.DataFrame(base_ref.T, columns=keys_ref)
ABX = pd.DataFrame(abx.T, columns=keys)
baseline = pd.DataFrame(base.T, columns=keys)
method = "Jaccard"
timepoints = 1
iters = 10
new = False
SIM = SimilarityCorrelation(ABX, baseline, post_ABX_container, baseline_ref, method, timepoints, iters, new, True)
sims_container = SIM.calc_similarity()
```

### Historical contingency
The Historical contingency analysis is designed to demonstrate the Historical contingency
mechanism using a theoretical model.
The Historical contingency analysis is implemented in the `historical_contingency.py` script.

#### Usage Example

```python
from methods.historical_contingency import HC

# Define the parameters
num_samples = 20
pool_size = 50
num_survived_min = 25
num_survived_max = 25
mean = 0
sigma = 5
c = 0.05
delta = 1e-5
final_time = 1000
max_step = 0.05
epsilon = 1e-4
phi = 1e-4
min_growth = 1
max_growth = 1
symmetric = True
alpha = None
method = 'RK45'
multiprocess = False
switch_off = False

# Calculate the results
H = HC(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta, final_time, max_step, epsilon,
       phi, min_growth, max_growth, symmetric, alpha, method, multiprocess)
results = H.get_results()
```

## Installation
Clone the project repository:
  ```sh
  git clone https://github.com/ShayaKahn/Recovery_from_antibiotics
  
