# Recovery of the microbiota after antibiotics treatment

This project contains a code library for the analysis of the recovery of the microbiota after antibiotic treatment using real and simulated data.
The project implements the null model, surrogate data analysis, similarity correlation analysis, and
historical contingency analysis. This code library is designed to derive insights into the recovery pattern of the gut
microbiota after antibiotic perturbation and the mechanisms related to this pattern using various
statistical techniques.  

## Methods

### Null model
The null model tests the following null hypothesis:

**H<sub>0</sub>**: The recovery pattern after antibiotic treatment is random sampling from the baseline distribution.

To test the null hypothesis for a given test subject we generate k random post-antibiotic treatment steady states where the species at each steady state are randomly sampled from a baseline distribution according 
to the species presence frequency in the baseline. By measuring k similarity values
between the baseline state of the test subject and the k random steady states, one can test the null hypothesis using a statistical test by comparing the k similarity values to the real similarity value between the baseline state and the post-antibiotic treatment state of the test subject.

The null model is implemented in the `null_model.py` script.

### Surrogate data analysis
The surrogate data analysis measures the similarity between the observed
post-antibiotics state of a test subject and the baseline state of other subjects
and the test subjects' baseline state. By comparing the similarity values to different subjects,
one can measure if the test subject shows a 'Personalized recovery' pattern. The surrogate data analysis is implemented
in the `surrogate.py` script.

### Similarity correlation
The similarity correlation analysis measures if there is a relation between new species
that appear after antibiotics treatment and the species that are considered as they survived the antibiotic treatment. This
analysis checks for evidence of the 'Historical contingency' mechanism.
The analysis is implemented in the `similarity_correlation.py` script.

### Historical contingency
The Historical contingency analysis is designed to demonstrate the Historical contingency
mechanism using a theoretical model.
The Historical contingency analysis is implemented in the `historical_contingency.py` script.


- Clone the project repository:
  ```sh
  git clone https://github.com/ShayaKahn/Recovery_from_antibiotics