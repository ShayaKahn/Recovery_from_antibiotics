from src.host_specific_recovery.metrics.dissimilarity import Dissimilarity
from src.host_specific_recovery.metrics.similarity import Similarity
import numpy as np

def run_similarity_analysis(dataset: dict) -> dict:

    baseline = dataset['baseline']
    post_ABX = dataset["post_abx_cohorts"][-1]

    unweighted_similarity_vector = []
    dissim_vector = []
    for base, post in zip(baseline, post_ABX):
        o = Similarity(base, post, method='Jaccard').calculate_similarity()
        d = Dissimilarity(base, post, method="rjsd").calculate_dissimilarity()
        unweighted_similarity_vector.append(o)
        dissim_vector.append(d)

    unweighted_similarity_vector = np.array(unweighted_similarity_vector)
    dissim_vector = np.array(dissim_vector)
    weighted_similarity_vector = 1. - dissim_vector

    return {
        "unweighted_similarity_vector": unweighted_similarity_vector,
        "weighted_similarity_vector": weighted_similarity_vector
    }
