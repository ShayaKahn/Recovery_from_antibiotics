import numpy as np

class ColonizationProbability:
    """
    This class is responsible in calculating the Cross-subjects colonization probabilities.
    """

    __slots__ = ("timeseries_tensor", "timeseries_tensor_binary", "T", "pattern_1_1", "pattern_1_0", "pattern_0_1",
                 "pattern_0_0", "pattern_1_1_mask", "pattern_1_0_mask", "pattern_0_1_mask", "pattern_0_0_mask",
                 "pattern_1_1_idx", "pattern_1_0_idx", "pattern_0_1_idx", "pattern_0_0_idx",
                 "pattern_1_1_idx_rel_abundances", "pattern_1_0_idx_rel_abundances", "pattern_0_1_idx_rel_abundances",
                 "pattern_0_0_idx_rel_abundances", "valid_taxa", "pattern_1_1_idx_flat_filtered",
                 "pattern_1_0_idx_flat_filtered", "pattern_0_1_idx_flat_filtered", "pattern_0_0_idx_flat_filtered",
                 "pattern_1_1_idx_rel_abundances_flat_filtered", "pattern_1_0_idx_rel_abundances_flat_filtered",
                 "pattern_0_1_idx_rel_abundances_flat_filtered", "pattern_0_0_idx_rel_abundances_flat_filtered")

    def __init__(self, timeseries_tensor):

        self.timeseries_tensor = timeseries_tensor
        self.timeseries_tensor_binary = (timeseries_tensor > 0).astype(int)
        self.T = timeseries_tensor.shape[0]
        self.pattern_1_1, self.pattern_1_0, self.pattern_0_1, self.pattern_0_0 = self._find_patterns()
        (self.pattern_1_1_mask, self.pattern_1_0_mask, self.pattern_0_1_mask,
         self.pattern_0_0_mask) = self._find_patterns_masks()
        (self.pattern_1_1_idx, self.pattern_1_0_idx, self.pattern_0_1_idx, self.pattern_0_0_idx,
         self.pattern_1_1_idx_rel_abundances, self.pattern_1_0_idx_rel_abundances, self.pattern_0_1_idx_rel_abundances,
         self.pattern_0_0_idx_rel_abundances) = self._find_patterns_indices()
        (self.valid_taxa, self.pattern_1_1_idx_flat_filtered, self.pattern_1_0_idx_flat_filtered,
         self.pattern_0_1_idx_flat_filtered, self.pattern_0_0_idx_flat_filtered,
         self.pattern_1_1_idx_rel_abundances_flat_filtered, self.pattern_1_0_idx_rel_abundances_flat_filtered,
         self.pattern_0_1_idx_rel_abundances_flat_filtered,
         self.pattern_0_0_idx_rel_abundances_flat_filtered) = self._find_filtered_patterns_indices()

    def _find_patterns(self):

        # Returned recolonized
        pattern_1_1 = np.hstack((np.array([1]), np.zeros(self.T - 3), np.array([1, 1])))[:, None, None]
        # Returned not recolonized
        pattern_1_0 = np.hstack((np.array([1]), np.zeros(self.T - 3), np.array([1, 0])))[:, None, None]
        # New colonized
        pattern_0_1 = np.hstack((np.array([0]), np.zeros(self.T - 3), np.array([1, 1])))[:, None, None]
        # New not colonized
        pattern_0_0 = np.hstack((np.array([0]), np.zeros(self.T - 3), np.array([1, 0])))[:, None, None]

        return pattern_1_1, pattern_1_0, pattern_0_1, pattern_0_0

    def _find_patterns_masks(self):

        # Returned recolonized mask
        pattern_1_1_mask = np.all(self.timeseries_tensor_binary == self.pattern_1_1, axis=0)
        # Returned not recolonized mask
        pattern_1_0_mask = np.all(self.timeseries_tensor_binary == self.pattern_1_0, axis=0)
        # New colonized mask
        pattern_0_1_mask = np.all(self.timeseries_tensor_binary == self.pattern_0_1, axis=0)
        # New not colonized mask
        pattern_0_0_mask = np.all(self.timeseries_tensor_binary == self.pattern_0_0, axis=0)

        return pattern_1_1_mask, pattern_1_0_mask, pattern_0_1_mask, pattern_0_0_mask

    def _find_patterns_indices(self):

        # Returned recolonized indices
        pattern_1_1_idx = [np.where(row)[0].tolist() for row in self.pattern_1_1_mask]
        pattern_1_1_idx_rel_abundances = [self.timeseries_tensor[-2, j, idx] for j, idx in enumerate(pattern_1_1_idx)]
        # Returned not recolonized indices
        pattern_1_0_idx = [np.where(row)[0].tolist() for row in self.pattern_1_0_mask]
        pattern_1_0_idx_rel_abundances = [self.timeseries_tensor[-2, j, idx] for j, idx in enumerate(pattern_1_0_idx)]
        # New colonized indices
        pattern_0_1_idx = [np.where(row)[0].tolist() for row in self.pattern_0_1_mask]
        pattern_0_1_idx_rel_abundances = [self.timeseries_tensor[-2, j, idx] for j, idx in enumerate(pattern_0_1_idx)]
        # New not colonized indices
        pattern_0_0_idx = [np.where(row)[0].tolist() for row in self.pattern_0_0_mask]
        pattern_0_0_idx_rel_abundances = [self.timeseries_tensor[-2, j, idx] for j, idx in enumerate(pattern_0_0_idx)]

        return (pattern_1_1_idx, pattern_1_0_idx, pattern_0_1_idx, pattern_0_0_idx,
                pattern_1_1_idx_rel_abundances, pattern_1_0_idx_rel_abundances,
                pattern_0_1_idx_rel_abundances, pattern_0_0_idx_rel_abundances)

    def _find_filtered_patterns_indices(self):
        # Flatten indices
        pattern_1_1_idx_flat = np.hstack(self.pattern_1_1_idx).flatten().astype(int)
        pattern_1_0_idx_flat = np.hstack(self.pattern_1_0_idx).flatten().astype(int)
        pattern_0_1_idx_flat = np.hstack(self.pattern_0_1_idx).flatten().astype(int)
        pattern_0_0_idx_flat = np.hstack(self.pattern_0_0_idx).flatten().astype(int)

        # Flatten relative abundances
        pattern_1_1_idx_rel_abundances_flat = np.hstack(self.pattern_1_1_idx_rel_abundances).flatten()
        pattern_1_0_idx_rel_abundances_flat = np.hstack(self.pattern_1_0_idx_rel_abundances).flatten()
        pattern_0_1_idx_rel_abundances_flat = np.hstack(self.pattern_0_1_idx_rel_abundances).flatten()
        pattern_0_0_idx_rel_abundances_flat = np.hstack(self.pattern_0_0_idx_rel_abundances).flatten()

        # Find the full group of new and returned taxa
        pattern_1_idx_flat = np.unique(np.hstack((pattern_1_1_idx_flat, pattern_1_0_idx_flat)).astype(int))
        pattern_0_idx_flat = np.unique(np.hstack((pattern_0_1_idx_flat, pattern_0_0_idx_flat)).astype(int))

        # Valid taxa are those that are both new and returned, we don't consider taxa that are only in one group
        valid_taxa = np.intersect1d(pattern_1_idx_flat, pattern_0_idx_flat)

        # Filter patterns to only include valid taxa
        pattern_1_1_idx_flat_filtered = pattern_1_1_idx_flat[np.isin(pattern_1_1_idx_flat, valid_taxa)]
        pattern_1_0_idx_flat_filtered = pattern_1_0_idx_flat[np.isin(pattern_1_0_idx_flat, valid_taxa)]
        pattern_0_1_idx_flat_filtered = pattern_0_1_idx_flat[np.isin(pattern_0_1_idx_flat, valid_taxa)]
        pattern_0_0_idx_flat_filtered = pattern_0_0_idx_flat[np.isin(pattern_0_0_idx_flat, valid_taxa)]

        # Filter relative abundances to only include valid taxa
        pattern_1_1_idx_rel_abundances_flat_filtered = pattern_1_1_idx_rel_abundances_flat[np.isin(pattern_1_1_idx_flat,
                                                                                                   valid_taxa)]
        pattern_1_0_idx_rel_abundances_flat_filtered = pattern_1_0_idx_rel_abundances_flat[np.isin(pattern_1_0_idx_flat,
                                                                                                   valid_taxa)]
        pattern_0_1_idx_rel_abundances_flat_filtered = pattern_0_1_idx_rel_abundances_flat[np.isin(pattern_0_1_idx_flat,
                                                                                                   valid_taxa)]
        pattern_0_0_idx_rel_abundances_flat_filtered = pattern_0_0_idx_rel_abundances_flat[np.isin(pattern_0_0_idx_flat,
                                                                                                   valid_taxa)]

        return (valid_taxa, pattern_1_1_idx_flat_filtered, pattern_1_0_idx_flat_filtered, pattern_0_1_idx_flat_filtered,
                pattern_0_0_idx_flat_filtered, pattern_1_1_idx_rel_abundances_flat_filtered,
                pattern_1_0_idx_rel_abundances_flat_filtered, pattern_0_1_idx_rel_abundances_flat_filtered,
                pattern_0_0_idx_rel_abundances_flat_filtered)

    def calc_probs(self):

        # Calculate probabilities and corresponding mean relative abundances
        probs = {}
        abundances = {}

        for taxa in self.valid_taxa:
            count_1_1 = np.sum(self.pattern_1_1_idx_flat_filtered == taxa)
            rel_abundance_1_1 = self.pattern_1_1_idx_rel_abundances_flat_filtered[
                self.pattern_1_1_idx_flat_filtered == taxa]
            if rel_abundance_1_1.size > 0:
                mean_rel_abundance_1_1 = rel_abundance_1_1.mean()
            else:
                mean_rel_abundance_1_1 = np.nan
            count_1_0 = np.sum(self.pattern_1_0_idx_flat_filtered == taxa)
            rel_abundance_1_0 = self.pattern_1_0_idx_rel_abundances_flat_filtered[
                self.pattern_1_0_idx_flat_filtered == taxa]
            if rel_abundance_1_0.size > 0:
                mean_rel_abundance_1_0 = rel_abundance_1_0.mean()
            else:
                mean_rel_abundance_1_0 = np.nan
            count_0_1 = np.sum(self.pattern_0_1_idx_flat_filtered == taxa)
            rel_abundance_0_1 = self.pattern_0_1_idx_rel_abundances_flat_filtered[
                self.pattern_0_1_idx_flat_filtered == taxa]
            if rel_abundance_0_1.size > 0:
                mean_rel_abundance_0_1 = rel_abundance_0_1.mean()
            else:
                mean_rel_abundance_0_1 = np.nan
            count_0_0 = np.sum(self.pattern_0_0_idx_flat_filtered == taxa)
            rel_abundance_0_0 = self.pattern_0_0_idx_rel_abundances_flat_filtered[
                self.pattern_0_0_idx_flat_filtered == taxa]
            if rel_abundance_0_0.size > 0:
                mean_rel_abundance_0_0 = rel_abundance_0_0.mean()
            else:
                mean_rel_abundance_0_0 = np.nan

            prob_1 = count_1_1 / (count_1_1 + count_1_0)
            prob_0 = count_0_1 / (count_0_1 + count_0_0)

            probs[taxa] = (prob_1, prob_0)

            abundances[taxa] = ((mean_rel_abundance_1_1, mean_rel_abundance_1_0),
                                (mean_rel_abundance_0_1, mean_rel_abundance_0_0))

        return probs, abundances