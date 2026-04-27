import numpy as np

def run_survived_species_analysis(dataset: dict, scale=1e2) -> dict:

    baseline = dataset["baseline"]
    ABX = dataset["abx"]
    posts = [np.asarray(P, float) for P in dataset["post_abx_cohorts"]]

    m, _ = baseline.shape
    results = []

    for i in range(m):
        base = baseline[i, :]
        abx = ABX[i, :]
        post_rows = [P[i, :] for P in posts]

        mask = (base > 0) & (abx > 0)
        for p in post_rows:
            mask &= (p > 0)

        mask &= (base / abx > scale)

        s = np.where(mask)[0]
        if s.size:
            measurements = [base[s], abx[s], *[p[s] for p in post_rows]]
            for j in range(np.size(measurements[0])):
                results.append(np.log10([val[j] for val in measurements]))

    results_matrix = np.vstack(results)
    mean = np.mean(results_matrix, axis=0)

    return {
        "results_matrix": results_matrix,
        "mean": mean
    }


