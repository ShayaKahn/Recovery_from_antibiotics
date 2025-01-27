from methods.similarity import Similarity
import numpy as np
import operator
from data_processing.optimal import OptimalCohort

class Surrogate:

    # This class implements Surrogate data analysis

    def __init__(self, base_samples_collections, test_base_sample, test_post_ABX_matrix, test_ABX_sample, timepoints=0,
                 strict=True, naive=False, method_opt='jaccard'):

        # Inputs:
        # base_samples_collections: dictionary of numpy vectors of shape (# species, ) or numpy matrices of shape
        #                           (# samples, # species) that represent the baseline of different subjects in the
        #                            experiment. The keys are the identifiers of the subjects.
        # test_base_sample: dictionary that contain a numpy vector of shape (# species, ) that represents the
        #                   baseline sample of a test subject. The key is subjects identifier.
        # test_post_ABX_matrix: numpy matrix of shape (# samples, # species) that contains the post antibiotics samples of
        #                       the test subject ordered chronologically in the rows.
        # test_ABX_sample: numpy array of shape (# species, ) that represents the antibiotics sample of the test subject.
        # timepoints: Integer, the number of time points after antibiotics administration the returned species considered
        #                      as survived. If dormant is False, timepoints have no meaning (None is the default).
        # strict: Boolean, if True, the returned species at time t are the species that are present in the baseline,
        #         absent in the antibiotic treatment and present in all the post antibiotic samples where t >= timepoints.
        #         If False, the returned species are the same except that they are present in time t = timepoints and can
        #         be absent in the post antibiotic samples at time t > timepoints except of the last sample in
        #          test_post_ABX_matrix.
        # naive: Boolean, if True, the subset of the species is all the species.
        # method_opt: String, the method used to choose the optimal samples. Choose from 'jaccard' and 'braycurtis'.

        (self.test_key, self.test_base_sample, self.base_samples_collections, self.test_post_ABX_matrix,
         self.test_ABX_sample) = Surrogate._validate_matrix_input(base_samples_collections, test_base_sample,
                                                                  test_post_ABX_matrix, test_ABX_sample)
        self.test_post_ABX_sample = self.test_post_ABX_matrix[-1, :]
        self.timepoints, self.strict, self.naive, method_opt = self._validate_input_variables(timepoints, strict, naive,
                                                                                              method_opt)
        if self.naive is True:
            self.subset = np.arange(self.test_base_sample.shape[0])
        else:
            self.subset = self._find_subset()

    @staticmethod
    def _validate_matrix_input(base_samples_collections, test_base_sample, test_post_ABX_matrix, test_ABX_sample):
        # Check if test_base_sample is in the correct type.
        if not (isinstance(test_base_sample, dict)):
            raise ValueError("test_base_sample must be dictionary")
        # separate the key and the matrix from the dictionary.
        if next(iter(test_base_sample.items()))[1].squeeze().ndim == 1:
            test_key, test_base_smp = next(iter(test_base_sample.items()))
        else:
            opt = OptimalCohort(test_base_sample, method='jaccard', norm=False)
            _, chosen_indices = opt.get_optimal_samples()
            test_key, chosen_index = next(iter(chosen_indices.items()))
            print(test_base_sample[test_key][chosen_index, :])
            test_base_smp = test_base_sample[test_key][chosen_index, :]
        # Check if the key is a string and the value is a numpy array.
        if not (isinstance(test_key, str) and isinstance(test_base_smp, np.ndarray)):
            raise ValueError("test_base_sample must be dictionary with string as a key and numpy array of"
                             " shape (# species, ) as value.")
        if not (isinstance(test_post_ABX_matrix, np.ndarray) and test_post_ABX_matrix.ndim == 2 and
                test_post_ABX_matrix.shape[1] == test_post_ABX_matrix.shape[1]):
            raise TypeError("test_post_ABX_matrix must be a 2D numpy array with the same number of columns as"
                            "test_base_sample and test_ABX_sample.")
        if not isinstance(test_ABX_sample, np.ndarray):
            raise ValueError("test_ABX_sample must be numpy array of shape (# species, ) as value.")
        # remove redundant dimensions.
        test_base_smp = test_base_smp.squeeze()
        # Check if base_samples_collections is in the correct type.
        if not (isinstance(base_samples_collections, dict)):
            raise ValueError("base_samples_collections must be dictionary")
        base_smp_collections = base_samples_collections
        # Check if the keys are strings and the values are numpy arrays.
        if not (all(isinstance(key, str) for key in base_smp_collections.keys())
                and all(isinstance(base_smp_collections[key],
                                   np.ndarray) for key in base_smp_collections.keys())):
            raise ValueError("base_samples_collections must be dictionary with string as keys and numpy array of"
                             " shape (# species, ) or (# samples, # species) as values.")
        dims = [base_smp_collections[key].shape[1] for key in base_smp_collections]
        dims.append(np.size(test_base_smp))
        dims.append(test_post_ABX_matrix.shape[1])
        dims.append(np.size(test_ABX_sample))
        if not all(dim == dims[0] for dim in dims):
            raise ValueError("The dimensions are not consistent across samples.")
        return test_key, test_base_smp.squeeze(), base_smp_collections, test_post_ABX_matrix, test_ABX_sample.squeeze()

    def _validate_input_variables(self, timepoints, strict, naive, method_opt):
        # Check if timepoints is integer.
        if not (isinstance(timepoints, int) and 0 <= timepoints < self.test_post_ABX_matrix.shape[0]):
            raise ValueError("timepoints must be integer greater then 0 and smaller then the number of samples in"
                             "test_post_ABX_matrix if dormant is True")
        if not (isinstance(strict, bool)):
            raise ValueError("strict must be boolean")
        if not (isinstance(naive, bool)):
            raise ValueError("naive must be boolean")
        if not (method_opt in ['jaccard', 'braycurtis']):
            raise ValueError("Invalid method. Choose from 'jaccard' and 'braycurtis'")
        return timepoints, strict, naive, method_opt

    def _find_subset(self):
        # Find the subset of the species should be removed during distance calculations.
        # Return:
        # indexes_comp: indexes of the species that should be removed.
        # indexes: indexes of the species that should be kept.

        # find the survived species.
        survived = (self.test_base_sample != 0) & (self.test_ABX_sample != 0) & (self.test_post_ABX_sample != 0)
        # find the resistant species.
        resistant = (self.test_base_sample == 0) & (self.test_ABX_sample != 0) & (self.test_post_ABX_sample != 0)
        # find the returned species.
        returned = self._returned_species()
        # find the subset of the species.
        conditions = [survived, resistant] + returned[:self.timepoints]
        all_groups = np.logical_or.reduce(conditions)
        indexes = np.where(~all_groups)[0]
        return indexes

    def apply_surrogate_data_analysis(self, method="Jaccard"):
        # calculate results using Similarity class
        results = {}
        # Calculate unweighted specificity/ recovery
        measure = Similarity(self.test_base_sample[self.subset], self.test_post_ABX_sample[self.subset],
                             method=method).calculate_similarity()
        # store the results of the test subject
        results[self.test_key] = measure
        for key in self.base_samples_collections.keys():
            if self.base_samples_collections[key].ndim == 1:
                measure_surrogate = Similarity(self.base_samples_collections[key][self.subset],
                                               self.test_post_ABX_sample[self.subset],
                                               method=method).calculate_similarity()
            else:
                measure_surrogate = np.mean(Similarity(self.test_post_ABX_sample[self.subset],
                                                        self.base_samples_collections[key][:, self.subset],
                                                        method=method).calculate_similarity())
            results[key] = measure_surrogate
        return results

    def _returned_species(self):
        num_timepoints = self.test_post_ABX_matrix.shape[0]
        op_lst = [operator.ne] * num_timepoints
        general_cond = [self.test_base_sample != 0, self.test_ABX_sample == 0, self.test_post_ABX_sample != 0]
        timepoints_vals = []
        if self.strict:
            for j in range(num_timepoints):
                inter_cond = [op_lst[i](self.test_post_ABX_matrix[i, :], 0) for i in range(j + 1)]
                special_cond = [op_lst[i](self.test_post_ABX_matrix[i, :], 0) for i in range(j + 1, num_timepoints)]
                cond = general_cond + inter_cond + special_cond
                timepoints_vals.append(np.logical_and.reduce(cond))
                if j != num_timepoints - 1:
                    op_lst[j] = operator.eq
        else:
            # iterate over the time points.
            for j in range(num_timepoints):
                # define the intermediate condition.
                inter_cond = [op_lst[i](self.test_post_ABX_matrix[i, :], 0) for i in range(j + 1)]
                # combine the conditions.
                cond = general_cond + inter_cond
                # find the returned species at each time point.
                timepoints_vals.append(np.logical_and.reduce(cond))
                if j != num_timepoints - 1:
                    # update the operators list.
                    op_lst[j] = operator.eq
        return timepoints_vals