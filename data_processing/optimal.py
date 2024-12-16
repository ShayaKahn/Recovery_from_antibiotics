from scipy.spatial.distance import pdist, squareform
import numpy as np

class OptimalCohort:
    """
    This Class generates Optimal cohort by choosing the optimal sample for each subject by pre-defined criterion.
    """
    def __init__(self, samples_dict, criterion='lower', method='braycurtis', norm=True):
        """
        samples_dict: dictionary that contains n subjects as a keys and a matrix of baseline samples, the rows of
                      the matrices represent the samples and the columns represent the species.
        criterion: the criterion to choose the optimal sample. Choose from 'lower' and 'mean'.
        """
        self.samples_dict, self.criterion, self.method = OptimalCohort._validate_input(samples_dict, criterion, method)
        if norm:
            self._normalize_data()
        self.dissimilarity_matrix_dict = self._create_dissimilarity_matrix_dict()

    @staticmethod
    def _validate_input(samples_dict, criterion, method):
        if not isinstance(samples_dict, dict):
            raise TypeError("samples_dict must be a dictionary")
        if not all(isinstance(key, str) for key in samples_dict.keys()):
            raise TypeError("The keys of samples_dict must be strings")
        if not all(isinstance(value, np.ndarray) for value in samples_dict.values()):
            raise TypeError("The values of samples_dict must be numpy arrays")
        if criterion not in ['lower', 'mean']:
            raise ValueError("Invalid criterion. Choose from 'lower' and 'mean'")
        dims = [mat.shape[1] for mat in samples_dict.values()]
        if not all(dim == dims[0] for dim in dims):
            raise ValueError("The dimensions are not consistent across samples.")
        if method not in ['braycurtis', 'jaccard']:
            raise ValueError("Invalid method. Choose from 'braycurtis' and 'jaccard'")
        return samples_dict, criterion, method

    def _create_dissimilarity_matrix_dict(self):
        """
        Create a dissimilarity matrix for each subject's baseline matrix.
        Return:
        dissimilarity_matrix_dict: dictionary, keys are the subjects and values are the dissimilarity matrices.
        """
        # Calculate dissimilarity matrix for each subject's baseline matrix
        dissimilarity_matrix_dict = {}
        # iterate over the subjects
        for key in self.samples_dict:
            # calculate the dissimilarity matrix
            y = pdist(self.samples_dict[key], metric='braycurtis')
            dissimilarity_matrix = squareform(y)
            dissimilarity_matrix_dict[key] = dissimilarity_matrix
        return dissimilarity_matrix_dict

    def _get_optimal_index(self):
        """
        Get the index of the chosen sample for each subject.
        Return:
        ind_container: dictionary, keys are the subjects and values are the indices of the chosen samples.
        """
        # get the index of the chosen sample for each subject
        ind_container = {}

        if self.criterion == 'lower':

            for key in self.dissimilarity_matrix_dict:
                # Get the indices of the lower triangle of the matrix
                ind = np.tril_indices(self.dissimilarity_matrix_dict[key].shape[0], k=-1, m=None)
                # Find the index of the smallest value in the lower triangle
                index = np.argmin(self.dissimilarity_matrix_dict[key][ind])
                # Convert the flattened index to row and column indices of the original matrix
                row_index, col_index = ind[0][index], ind[1][index]
                indices = (row_index, col_index)
                ind_container[key] = indices
            return ind_container

        elif self.criterion == 'mean':

            for key in self.dissimilarity_matrix_dict:
                # calculate the mean distance for each sample
                mean_dist = np.mean(self.dissimilarity_matrix_dict[key], axis=1)
                # get the index of the sample with the minimum mean distance
                min_idx = np.argmin(mean_dist)
                ind_container[key] = min_idx
            return ind_container

    def get_optimal_samples(self):
        # get the optimal samples for each subject with the corresponding index
        ind_container = self._get_optimal_index()
        optimal_samples = []
        chosen_indices = {}

        if self.criterion == 'lower':

            for key in self.samples_dict:
                # Calculate the mean dissimilarity of each row to the other rows and choose the row with the lower mean
                mean_dissimilarity_row0 = np.mean(self.dissimilarity_matrix_dict[key][ind_container[key][0], :])
                mean_dissimilarity_row1 = np.mean(self.dissimilarity_matrix_dict[key][ind_container[key][1], :])
                if mean_dissimilarity_row0 <= mean_dissimilarity_row1:
                    optimal_samples.append(self.samples_dict[key][ind_container[key][0], :])
                    chosen_indices[key] = ind_container[key][0]
                else:
                    optimal_samples.append(self.samples_dict[key][ind_container[key][1], :])
                    chosen_indices[key] = ind_container[key][1]
            optimal_samples = np.vstack(optimal_samples)
            return optimal_samples, chosen_indices

        elif self.criterion == 'mean':

            for key in self.samples_dict:
                optimal_samples.append(self.samples_dict[key][ind_container[key], :])
                chosen_indices[key] = ind_container[key]
            optimal_samples = np.vstack(optimal_samples)
            return optimal_samples, chosen_indices

    def _normalize_data(self):
        for key in self.samples_dict:
            # Normalize the samples to sum to 1
            self.samples_dict[key] = self.samples_dict[key] / \
                                     self.samples_dict[key].sum(axis=1, keepdims=True)