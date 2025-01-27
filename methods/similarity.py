import numpy as np
from skbio.diversity import beta_diversity

class Similarity:
    # This class calculates different similarity measures between two given samples or between sample and matrix.

    def __init__(self, sample_first, matrix, method="Overlap", norm=True):
        # Inputs:
        # sample_first: first sample, 1D array of shape (# species,) or 1D array of shapes (1, # species), (# species, 1).
        # matrix: matrix, 1D array of shape (# species,) or 1D array of shapes (1, # species), (# species, 1)
        # or matrix of shape (# samples, # species).
        # method: similarity measure to be calculated. Default is "Overlap".
        # norm: boolean, if True the samples will be normalized. Default is True.

        self.sample_first, self.matrix, self.method = self._validate_input(sample_first, matrix, method)
        # Normalize the samples.
        if not isinstance(norm, bool):
            raise TypeError("norm must be a boolean")
        if norm:
            self.sample_first, self.matrix = self._normalize()

    def _validate_input(self, sample_first, matrix, method):
        # Inputs:
        # As defined in the __init__ method.
        # Returns:
        # sample_first: first sample, 1D array of shape (# species,).
        # matrix: matrix, 1D array of shape (# species,) or matrix of shape (# samples, # species).
        # method: As defined in the __init__ method.

        # Check if the inputs are numpy arrays.
        if not isinstance(sample_first, np.ndarray) or not isinstance(matrix, np.ndarray):
            raise TypeError("sample_first and sample_second must be numpy arrays")
        # Check if sample first is one dimensional and matrix is two-dimensional or one dimensional.
        if sample_first.ndim == 2:
            sample_first = sample_first.squeeze()
        if matrix.ndim == 2 and matrix.shape[0] == 1:
            matrix = matrix.squeeze()
        if not ((sample_first.ndim == 1) and (matrix.ndim == 1 or matrix.ndim == 2)):
            raise TypeError("sample_first should be a 1D array and matrix should be a 1D or 2D array")
        # Check if the samples have the same length.
        if matrix.ndim == 2:
            if sample_first.size != matrix.shape[1]:
                raise ValueError("sample_first and the rows in matrix must have the same length")
        elif matrix.ndim == 1:
            if sample_first.size != matrix.size:
                raise ValueError("sample_first and matrix must have the same length")
        if method not in ["Overlap", "Jaccard", "Dice", "Weighted Jaccard", "Szymkiewicz Simpson", "Recovery",
                          "Specificity",
                          "Unweighted Unifrac", "Weighted Jaccard symmetric"]:
            raise ValueError("Invalid method. Choose from: 'Overlap', 'Jaccard', 'Dice', 'Weighted Jaccard',"
                             " Weighted Jaccard symmetric", 'Szymkiewicz Simpson', 'Recovery',
                             " 'Specificity', and 'Unweighted Unifrac'")
        if np.any(np.isnan(matrix)):
            raise ValueError("matrix contains NaN values")
        if np.any(np.isnan(sample_first)):
            raise ValueError("sample_first contains NaN values")
        return sample_first, matrix, method

    def _normalize(self):
        sum_first = np.sum(self.sample_first)
        if sum_first == 0:
            normalized_sample_first = self.sample_first
        else:
            normalized_sample_first = self.sample_first / sum_first
        if self.matrix.ndim == 1:
            sum_mat = np.sum(self.matrix)
            if sum_mat == 0:
                normalized_mat = self.matrix
            else:
                normalized_mat = self.matrix / sum_mat
        elif self.matrix.ndim == 2:
            sum_mat = np.sum(self.matrix, axis=1)
            mask = (sum_mat == 0)
            try:
                assert not np.any(mask)
                normalized_mat = self.matrix / sum_mat[..., None]
            except AssertionError:
                print('One of the samples in the matrix is empty')
                normalized_mat = self.matrix / sum_mat[..., None]
                idx = np.where(mask)
                normalized_mat[idx, :] = self.matrix[idx, :]
        return normalized_sample_first, normalized_mat

    @staticmethod
    def _find_intersection(v1, v2):
        # This method finds the shared non-zero indexes of the two samples.
        # Inputs:
        # v1: one dimentional numpy array
        # v2: one dimentional numpy array
        # Returns:
        # The set s with represent the intersected indexes

        nonzero_index_first = np.nonzero(v1)  # Find the non-zero index of the first sample.
        nonzero_index_second = np.nonzero(v2)  # Find the non-zero index of the second sample.
        s = np.intersect1d(nonzero_index_first, nonzero_index_second)  # Find the intersection.
        return s

    def _overlap(self):
        # This method calculates the overlap between the first sample and the matrix.
        # Rerurns:
        # if the matrix is 1D: overlap value.
        # if the matrix is 2D: numpy array that contain the overlap values.

        if self.matrix.ndim == 1:
            # find the intersection
            s = self._find_intersection(self.sample_first, self.matrix)
            # calculate the overlap
            overlap = np.sum(self.sample_first[s] + self.matrix[s]) / 2
            return overlap
        elif self.matrix.ndim == 2:
            overlaps = []
            for smp in self.matrix:
                # find the intersection
                s = self._find_intersection(self.sample_first, smp)
                # calculate the overlap
                overlap = np.sum(self.sample_first[s] + smp[s]) / 2
                overlaps.append(overlap)
            return np.array(overlaps)

    def _jaccard(self):
        # This method calculates the Jaccard similarity between the first sample and the matrix.
        # Returns:
        # if the matrix is 1D: Jaccard similarity value.
        # if the matrix is 2D: numpy array that contain the Jaccard similarity values.

        sample_first_boolean = np.where(self.sample_first != 0, 1, 0)
        matrix_boolean = np.where(self.matrix != 0, 1, 0)
        # find the intersection and the union
        intersection_bool = np.logical_and(sample_first_boolean, matrix_boolean)
        union_bool = np.logical_or(sample_first_boolean, matrix_boolean)
        if self.matrix.ndim == 1:
            # calculate the Jaccard similarity
            intersection = np.sum(intersection_bool)
            union = np.sum(union_bool)
            jaccard = intersection / union
            return jaccard
        elif self.matrix.ndim == 2:
            # calculate the Jaccard similarity
            intersection = np.sum(intersection_bool, axis=1)
            union = np.sum(union_bool, axis=1)
            jaccard = intersection / union
            return jaccard

    def _dice(self):
        # This method calculates the Dice similarity between the first sample and the matrix.
        # Returns:
        # if the matrix is 1D: Dice similarity value.
        # if the matrix is 2D: numpy array that contain the Dice similarity values.

        sample_first_boolean = np.where(self.sample_first != 0, 1, 0)
        matrix_boolean = np.where(self.matrix != 0, 1, 0)
        # find the intersection
        intersection_bool = np.logical_and(sample_first_boolean, matrix_boolean)
        if self.matrix.ndim == 1:
            # calculate the Dice similarity
            intersection = np.sum(intersection_bool)
            dice = 2 * intersection / (sample_first_boolean.sum() + matrix_boolean.sum())
            return dice
        elif self.matrix.ndim == 2:
            # calculate the Dice similarity
            intersection = np.sum(intersection_bool, axis=1)
            dice = 2 * intersection / (sample_first_boolean.sum() + matrix_boolean.sum(axis=1))
            return dice

    def _szymkiewicz_simpson(self):
        # This method calculates the Szymkiewicz Simpson similarity between the first sample and the matrix.
        # Returns:
        # if the matrix is 1D: Szymkiewicz Simpson similarity value.
        # if the matrix is 2D: numpy array that contain the Szymkiewicz Simpson similarity values.

        sample_first_boolean = np.where(self.sample_first != 0, 1, 0)
        matrix_boolean = np.where(self.matrix != 0, 1, 0)
        # find the intersection
        intersection_bool = np.logical_and(sample_first_boolean, matrix_boolean)
        if self.matrix.ndim == 1:
            # calculate the Szymkiewicz Simpson similarity
            denom = np.min((sample_first_boolean.sum(), matrix_boolean.sum()))
            intersection = np.sum(intersection_bool)
            ss = intersection / denom
            return ss
        elif self.matrix.ndim == 2:
            # calculate the Szymkiewicz Simpson similarity
            denom = np.array([np.min((sample_first_boolean.sum(), smp.sum())) for smp in matrix_boolean])
            intersection = np.sum(intersection_bool, axis=1)
            ss = intersection / denom
            return ss

    def _recovery(self):
        # This method calculates the Recovery similarity between the first sample and the matrix.
        # Returns:
        # if the matrix is 1D: Recovery similarity value.
        # if the matrix is 2D: numpy array that contain the Recovery similarity values.

        sample_first_boolean = np.where(self.sample_first != 0, 1, 0)
        matrix_boolean = np.where(self.matrix != 0, 1, 0)
        # find the intersection
        intersection_bool = np.logical_and(sample_first_boolean, matrix_boolean)
        # find the size of the first sample
        nonzero_size_first = sample_first_boolean.sum()
        if self.matrix.ndim == 1:
            # calculate the Recovery similarity
            intersection = np.sum(intersection_bool)
            recovery = intersection / nonzero_size_first
            return recovery
        elif self.matrix.ndim == 2:
            # calculate the Recovery similarity
            intersection = np.sum(intersection_bool, axis=1)
            recovery = intersection / nonzero_size_first
            return recovery

    def _specificity(self):
        # This method calculates the Specificity similarity between the first sample and the matrix.
        # Returns:
        # if the matrix is 1D: Specificity similarity value.
        # if the matrix is 2D: numpy array that contain the Specificity similarity values.

        sample_first_boolean = np.where(self.sample_first != 0, 1, 0)
        matrix_boolean = np.where(self.matrix != 0, 1, 0)
        # find the intersection
        intersection_bool = np.logical_and(sample_first_boolean, matrix_boolean)
        if self.matrix.ndim == 1:
            # calculate the Specificity similarity
            nonzero_size_mat = matrix_boolean.sum()
            intersection = np.sum(intersection_bool)
            specificity = intersection / nonzero_size_mat
            return specificity
        elif self.matrix.ndim == 2:
            # calculate the Specificity similarity
            nonzero_size_mat = matrix_boolean.sum(axis=1)
            intersection = np.sum(intersection_bool, axis=1)
            specificity = intersection / nonzero_size_mat
            return specificity

    def _weighted_jaccard(self):
        # This method calculates the Weighted Jaccard similarity between the first sample and the matrix.
        # Returns:
        # if the matrix is 1D: Weighted Jaccard similarity value.
        # if the matrix is 2D: numpy array that contain the Weighted Jaccard similarity values.

        # find the intersection and the union
        sample_first_boolean = np.where(self.sample_first != 0, 1, 0)
        matrix_boolean = np.where(self.matrix != 0, 1, 0)
        intersection_bool = np.logical_and(sample_first_boolean, matrix_boolean)
        union_bool = np.logical_or(sample_first_boolean, matrix_boolean)
        if self.matrix.ndim == 1:
            # calculate the Weighted Jaccard similarity
            sum_first_intersect = np.sum(self.sample_first[intersection_bool])
            sum_first_union = np.sum(self.sample_first[union_bool])
            jaccard_w = sum_first_intersect / sum_first_union
            return jaccard_w
        elif self.matrix.ndim == 2:
            # calculate the Weighted Jaccard similarity
            sum_first_intersect = np.array([self.sample_first[inter].sum() for inter in intersection_bool])
            sum_first_union = np.array([self.sample_first[un].sum() for un in union_bool])
            jaccard_w = sum_first_intersect / sum_first_union
            return jaccard_w

    def _weighted_jaccard_symmetric(self):
        # This method calculates the symmetric Weighted Jaccard similarity between the first sample and the matrix.
        # Returns:
        # if the matrix is 1D: symmetric Weighted Jaccard similarity value.
        # if the matrix is 2D: numpy array that contain the symmetric Weighted Jaccard similarity values.

        if self.matrix.ndim == 1:
            # calculate the symmetric Weighted Jaccard similarity
            min_sum = np.sum(np.minimum(self.sample_first, self.matrix))
            max_sum = np.sum(np.maximum(self.sample_first, self.matrix))
            jaccard_w_sym = min_sum / max_sum
            return jaccard_w_sym
        elif self.matrix.ndim == 2:
            # calculate the symmetric Weighted Jaccard similarity
            min_sum = np.sum(np.minimum(self.sample_first, self.matrix), axis=1)
            max_sum = np.sum(np.maximum(self.sample_first, self.matrix), axis=1)
            jaccard_w_sym = min_sum / max_sum
            return jaccard_w_sym

    def _unweighted_unifrac(self, tree, otu_ids):
        # This method calculates the Unweighted Unifrac similarity between the first sample and the matrix.
        # Inputs:
        # tree: phylogenetic tree object.
        # otu_ids: list of OTU ids.
        # Returns:
        # if the matrix is 1D: Unweighted Unifrac similarity value.
        # if the matrix is 2D: numpy array that contain the Unweighted Unifrac similarity values.

        if self.matrix.ndim == 1:
            # calculate the Unweighted Unifrac similarity
            data = np.vstack([self.sample_first, self.matrix])
            sample_ids = np.arange(0, data.shape[0], 1).tolist()
            uu = 1 - beta_diversity(metric='unweighted_unifrac', counts=data, ids=sample_ids, taxa=otu_ids,
                                    tree=tree, validate=False)[1, 0]
            return uu
        elif self.matrix.ndim == 2:
            # calculate the Unweighted Unifrac similarity
            uu = []
            for smp in self.matrix:
                data = np.vstack([self.sample_first, smp])
                sample_ids = np.arange(0, data.shape[0], 1).tolist()
                uu.append(1 - beta_diversity(metric='unweighted_unifrac', counts=data, ids=sample_ids, taxa=otu_ids,
                                             tree=tree, validate=False)[1, 0])
            return np.array(uu)
    def calculate_similarity(self, tree=None, otu_ids=None):
        # This method calculates the similarity between the first sample and the matrix.
        # Inputs:
        # tree: phylogenetic tree object. Default is None. Only used for Unweighted Unifrac.
        # otu_ids: list of OTU ids. Default is None. Only used for Unweighted Unifrac.
        # Returns:
        # if the matrix is 1D: similarity value.
        # if the matrix is 2D: numpy array that contain the similarity values.

        # Calculation of the similarity based on the method.
        if self.method == "Overlap":
            return self._overlap()
        elif self.method == "Jaccard":
            return self._jaccard()
        elif self.method == "Dice":
            return self._dice()
        elif self.method == "Szymkiewicz Simpson":
            return self._szymkiewicz_simpson()
        elif self.method == "Recovery":
            return self._recovery()
        elif self.method == "Specificity":
            return self._specificity()
        elif self.method == "Weighted Jaccard":
            return self._weighted_jaccard()
        elif self.method == "Weighted Jaccard symmetric":
            return self._weighted_jaccard_symmetric()
        elif self.method == "Unweighted Unifrac":
            return self._unweighted_unifrac(tree, otu_ids)