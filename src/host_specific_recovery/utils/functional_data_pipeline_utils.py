import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

def similarity_matrix(A: pd.DataFrame, B: pd.DataFrame, metric: str = "braycurtis",
                      fill_value: float = 0.0) -> pd.DataFrame:
    """
    :param A: A Pandas DataFrame, represent columns as samples and rows as features
    :param B: A Pandas DataFrame, represent columns as samples and rows as features
    :param metric: "braycurtis", "weighted_jaccard", or "Jaccard"
    :param fill_value: value to fill in missing rows when aligning A and B
    :return: DataFrame of shape (B.shape[1], A.shape[1]) representing pairwise similarities
    """

    metric = metric.lower()
    if metric not in {"braycurtis", "weighted_jaccard", "Jaccard"}:
        raise ValueError("metric must be one of: 'braycurtis', 'weighted_jaccard', 'Jaccard'")

    A2, B2 = A.align(B, join="outer", axis=0, fill_value=fill_value)

    X = A2.to_numpy(dtype=float, copy=False)
    Y = B2.to_numpy(dtype=float, copy=False)

    p = X.shape[1]
    q = Y.shape[1]
    sims = np.empty((q, p), dtype=float)

    if metric == "weighted_jaccard":
        for i in range(p):
            x = X[:, i][:, None]  # (n,1)
            num = np.minimum(x, Y).sum(axis=0)
            den = np.maximum(x, Y).sum(axis=0)
            sims[:, i] = np.divide(num, den, out=np.ones_like(num), where=den != 0)

    elif metric == "braycurtis":
        sumY = Y.sum(axis=0)
        for i in range(p):
            x = X[:, i][:, None]
            num = 2.0 * np.minimum(x, Y).sum(axis=0)
            den = x.sum(axis=0)[0] + sumY
            sims[:, i] = np.divide(num, den, out=np.ones_like(num), where=den != 0)

    else:  # Jaccard
        Xb = X > 0.0
        Yb = Y > 0.0
        for i in range(p):
            x = Xb[:, i][:, None]
            inter = np.logical_and(x, Yb).sum(axis=0)
            union = np.logical_or(x, Yb).sum(axis=0)
            sims[:, i] = np.divide(inter, union, out=np.ones_like(inter, dtype=float), where=union != 0)

    return pd.DataFrame(sims, index=B2.columns.astype(str), columns=A2.columns.astype(str))

def find_colonizers(base: Union[pd.Series, List[pd.Series]], abx: List[pd.Series], post_ABX_samples: List[pd.Series],
                    k: int=2) -> Tuple[set, set]:
    """
    :param base: The baseline sample, or a list of baseline samples to be concatenated
    :param abx: List of antibiotics samples
    :param post_ABX_samples: List of post-antibiotics samples
    :param k: The index of the target post-antibiotics sample to check for colonizers (1-based from the end)
    :return: A tuple containing two sets: the first set contains colonizers, and the second set contains transient.
    """
    if isinstance(base, pd.Series):
        base = [base]
    standard_mask = (pd.concat(base, axis=1).eq(0).all(axis=1) & pd.concat(abx, axis=1).eq(0).all(axis=1))

    target_transient = post_ABX_samples[-k]
    others_transient = pd.concat(post_ABX_samples[:len(post_ABX_samples) - k] + post_ABX_samples[len(
        post_ABX_samples) - k + 1:], axis=1)

    transient_mask = standard_mask & target_transient.gt(0) & others_transient.eq(0).all(axis=1)
    colonizers_mask = (post_ABX_samples[-2] > 0) & (post_ABX_samples[-1] > 0)

    colonizers = set(post_ABX_samples[-1].index[colonizers_mask])
    transient = set(post_ABX_samples[-k].index[transient_mask])

    return colonizers, transient

def two_sided_pvalues_bh_with_effect(pairs, alpha=0.05, decimals=6, min_n=5):
    """
    :param pairs: List of pairs of arrays/lists to compare [(x1, y1), (x2, y2), ...]
    :param alpha: Threshold for Benjamini-Hochberg correction
    :param decimals: Number of decimals to display in p-value strings
    :param min_n: Minimum sample size to perform the test; smaller samples are skipped
    :return: List of dictionaries with test results for each pair
    """
    thr = 10 ** (-decimals)

    def fmt(pv):
        if not np.isfinite(pv):
            return "nan"
        return f"{pv:.{decimals}f}" if pv >= thr else f"< {thr:.{decimals}f}"

    results = []
    pvals = []
    tested_idx = []

    for i, (x, y) in enumerate(pairs):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        # effect is still defined (nan if one side empty)
        effect = float(np.mean(x) - np.mean(y)) if (x.size and y.size) else np.nan

        # treat empty as "n < min_n" (skip), instead of raising
        if x.size < min_n or y.size < min_n:
            results.append({
                "test": "skipped (n < min_n)",
                "n_x": int(x.size),
                "n_y": int(y.size),
                "effect_mean_diff": effect,
                "auc": np.nan,
                "statistic": np.nan,
                "p_raw": np.nan,
                "p_raw_str": "nan",
                "q_bh": np.nan,
                "q_bh_str": "nan",
                "reject_bh": False,
            })
            continue

        # MWU p-value
        test = "Mann–Whitney U (two-sided)"
        stat, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        p = float(p)

        # AUC (AUC of predicting x vs y using the raw values as scores)
        scores = np.concatenate([x, y])
        y_true = np.concatenate([np.ones(x.size, dtype=int), np.zeros(y.size, dtype=int)])
        auc = float(roc_auc_score(y_true, scores))

        results.append({
            "test": test,
            "n_x": int(x.size),
            "n_y": int(y.size),
            "effect_mean_diff": float(np.mean(x) - np.mean(y)),
            "auc": auc,
            "statistic": float(stat),
            "p_raw": p,
            "p_raw_str": fmt(p),
        })
        pvals.append(p)
        tested_idx.append(len(results) - 1)

    # BH only over tests that were actually run
    if pvals:
        reject, qvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        for idx, qv, rej in zip(tested_idx, qvals, reject):
            results[idx]["q_bh"] = float(qv)
            results[idx]["q_bh_str"] = fmt(float(qv))
            results[idx]["reject_bh"] = bool(rej)
    else:
        for r in results:
            r.setdefault("q_bh", np.nan)
            r.setdefault("q_bh_str", "nan")
            r.setdefault("reject_bh", False)

    return results

def filter_valid_pairs(eff1, p1, eff2, p2, require_p_in_01: bool = True):
    eff1 = np.asarray(eff1, dtype=float).ravel()
    p1 = np.asarray(p1, dtype=float).ravel()
    eff2 = np.asarray(eff2, dtype=float).ravel()
    p2 = np.asarray(p2, dtype=float).ravel()

    lens = [eff1.size, p1.size, eff2.size, p2.size]
    if len(set(lens)) != 1:
        raise ValueError(f"Length mismatch: eff1={lens[0]}, p1={lens[1]}, eff2={lens[2]}, p2={lens[3]}")

    m = np.isfinite(eff1) & np.isfinite(p1) & np.isfinite(eff2) & np.isfinite(p2)
    if require_p_in_01:
        m &= (0.0 <= p1) & (p1 <= 1.0) & (0.0 <= p2) & (p2 <= 1.0)

    return eff1[m], p1[m], eff2[m], p2[m]
