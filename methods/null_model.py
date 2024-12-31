import numpy as np
from scipy.spatial.distance import braycurtis, jensenshannon
from methods.similarity import Similarity
from skbio.diversity import beta_diversity
from cython_modules.null_model_functions import generate_samples
import operator

class NullModel:
    """
    This class is responsible to create the null model for a test subject. Given a baseline sample,
     antibiotics sample and post antibiotics sample (post-antibiotics sample).
    """
    def __init__(self, test_base_sample, test_ABX_sample, baseline_cohort, test_post_ABX_matrix,
                 num_reals=100, timepoints=0, strict=True):
        """
        Inputs:
        test_base_sample: baseline sample of the test subject, numpy array of shape (# species,).
        test_ABX_sample: antibiotics sample of the test subject, numpy array of shape (# species,).
        baseline_cohort: baseline cohort, numpy array of shape (# samples, # species).
        test_post_ABX_matrix: numpy matrix of shape (# samples, # species) that contains the post
                              antibiotics samples of the test subject ordered chronologically in the rows.
        num_reals: number of realizations of the null model (number of synthetic samples).
        timepoints: Integer, the number of time points after antibiotics administration
                    the returned species considered as survived.
        strict: Boolean, if True, the returned species at time t are the species that are present in the baseline,
                absent in the antibiotic treatment and present in all the post antibiotic samples where t >= timepoints.
                If False, the returned species are the same except that they are present in time t = timepoints and can
                be absent in the post antibiotic samples at time t > timepoints except of the last sample in
                 test_post_ABX_matrix.
        """
        # validate the inputs.
        (self.test_base_sample, self.test_ABX_sample, self.baseline_cohort,
         self.test_post_ABX_matrix) = self._validate_input_matrices(test_base_sample, test_ABX_sample,
                                                                    baseline_cohort, test_post_ABX_matrix)
        # define the steady state after the antibiotics administration.
        self.test_post_ABX_sample = self.test_post_ABX_matrix[-1, :]
        self.num_reals, self.timepoints, self.strict = self._validate_input_variables(num_reals, timepoints, strict)
        # Find the subset of the chosen species.
        self.subset_comp, self.subset = self._find_subset()
        # Create the synthetic samples.
        self.synthetic_samples = self._create_synthetic_samples()

    def _validate_input_matrices(self, test_base_sample, test_ABX_sample, baseline_cohort, test_post_ABX_matrix):
        # Check if the inputs are numpy arrays with consistent dimensions.
        if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [test_base_sample, test_ABX_sample]):
            raise TypeError("test_base_sample and test_ABX_sample must be 1D numpy arrays.")
        if not (isinstance(baseline_cohort, np.ndarray) and baseline_cohort.ndim == 2 and
                baseline_cohort.shape[1] == np.size(test_base_sample)):
            raise TypeError("baseline_cohort must be a 2D numpy array with the same number of columns as"
                            "test_base_sample and ABX_sample.")
        if not (isinstance(test_post_ABX_matrix, np.ndarray) and test_post_ABX_matrix.ndim == 2 and
                test_post_ABX_matrix.shape[1] == np.size(test_post_ABX_matrix[0, :])):
            raise TypeError("test_post_ABX_matrix must be a 2D numpy array with the same number of columns as"
                            " baseline_sample and test_ABX_sample.")
        if not all(np.size(arr) == np.size(test_base_sample) for arr in [test_ABX_sample, test_post_ABX_matrix[0, :]]):
            raise ValueError("test_ABX_sample and test_post_ABX_matrix must have the same dimension as"
                             " test_base_sample.")
        # Check if the inputs are not empty.
        if 0 in [test_base_sample.sum(), test_ABX_sample.sum()]:
            raise ValueError("At least one of the input samples is empty")
        if 0 in baseline_cohort.sum(axis=1):
            raise ValueError("At least one of the samples in the baseline cohort is empty")
        if 0 in test_post_ABX_matrix.sum(axis=1):
            raise ValueError("At least one of the samples in the test_post_ABX_matrix is empty")
        # Normalize the inputs if necessary.
        if np.isclose(np.sum(test_base_sample), 1.0):
            base = test_base_sample.squeeze()
        else:
            base = self._normalize_cohort(test_base_sample.squeeze())
        if np.isclose(np.sum(test_ABX_sample), 1.0):
            abx = test_ABX_sample
        else:
            abx = self._normalize_cohort(test_ABX_sample)
        if np.isclose(np.sum(baseline_cohort.sum(axis=1)), baseline_cohort.shape[0]):
            base_cohort = baseline_cohort
        else:
            base_cohort = self._normalize_cohort(baseline_cohort)
        if np.isclose(np.sum(test_post_ABX_matrix.sum(axis=1)), test_post_ABX_matrix.shape[0]):
            post_cohort = test_post_ABX_matrix
        else:
            post_cohort = self._normalize_cohort(test_post_ABX_matrix)
        return base, abx, base_cohort, post_cohort

    def _validate_input_variables(self, num_reals, timepoints, strict):
        # Check if the number of realizations is valid.
        if not (isinstance(num_reals, int) and 0 < num_reals <= 10000):
            raise ValueError("num_reals must be an integer greater than 0 and smaller or equal to 10000.")
        # Check if timepoints is valid.
        if not (isinstance(timepoints, int)):
            raise ValueError("timepoints must be integer if dormant is True")
        if not (0 <= timepoints < self.test_post_ABX_matrix.shape[0]):
            raise ValueError("timepoints must be an integer greater then 0 and smaller then the number of rows in"
                             " test_post_ABX_matrix.")
        # Check if strict is valid.
        if not (isinstance(strict, bool)):
            raise ValueError("strict must be of type bool.")
        return num_reals, timepoints, strict

    def _find_subset(self):
        """
        Find the subset of the species should be removed during distance calculations.
        Return:
        indexes_comp: indexes of the species that should be removed.
        indexes: indexes of the species that should be kept.
        """
        # find the survived species.
        survived = (self.test_base_sample != 0) & (self.test_ABX_sample != 0) & (self.test_post_ABX_sample != 0)
        # find the resistant species.
        resistant = (self.test_base_sample == 0) & (self.test_ABX_sample != 0) & (self.test_post_ABX_sample != 0)
        # find the returned species.
        returned = self._returned_species()
        # combine the conditions.
        conditions = [survived, resistant] + returned[:self.timepoints]
        # find the subset of the species.
        all_groups = np.logical_or.reduce(conditions)
        indexes = np.where(~all_groups)[0]
        indexes_comp = np.setdiff1d(np.arange(np.size(self.test_base_sample)), indexes)
        return indexes_comp, indexes

    def _create_synthetic_samples(self):
        """
        Create the synthetic samples.
        Return:
        synthetic_samples, normalized numpy array of shape (# realizations, # species).
        """
        # set the constraint on the number of species
        stop = np.size(np.nonzero(self.test_post_ABX_sample[self.subset]))

        # define the pool of species to be shuffled not including the subset species
        # the pool contains the values where the baseline cohort is not zero
        pool = self.baseline_cohort[self.baseline_cohort != 0]
        # the pool_indices contains the column(species) indexes of the corresponding species in the pool
        pool_indices = np.where(self.baseline_cohort != 0)[1].astype(np.int32)

        # remove the subset species from the pool and the pool indices
        mask = np.isin(pool_indices, self.subset)
        pool_indices = pool_indices[mask]
        pool = pool[mask]
        synthetic_samples = generate_samples(pool, pool_indices, stop, self.num_reals,
                                             np.size(self.test_post_ABX_sample))
        # Normalize the synthetic_samples.
        synthetic_samples = self._normalize_cohort(synthetic_samples)
        return synthetic_samples

    @staticmethod
    def _normalize_cohort(cohort):
        # normalization function
        if cohort.ndim == 1:
            cohort_normalized = cohort / cohort.sum()
        else:
            cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
        return cohort_normalized

    def _bray_curtis(self):
        """
        Calculate the Bray Curtis distance between the test sample and the baseline sample and the Bray Curtis distance
        between the synthetic samples and the baseline sample.
        Return:
        dist_real: distance between the test sample and the baseline sample.
        dist_synthetic: distances between the shuffled samples and the baseline sample.
        """
        # Calculate the Bray Curtis distance between the test sample and the baseline sample.
        dist_real = braycurtis(self._normalize_cohort(self.test_base_sample[self.subset]),
                               self._normalize_cohort(self.test_post_ABX_sample[self.subset]))
        # Calculate the Bray Curtis distance between the synthetic samples and the baseline sample.
        dist_synthetic = np.apply_along_axis(lambda x: braycurtis(self._normalize_cohort(
            self.test_base_sample[self.subset]), x[self.subset]), 1, self.synthetic_samples)
        return dist_real, dist_synthetic

    def _rJSD(self):
        """
        Calculate the rJSD distance between the test sample and the baseline sample and the rJSD distance
        between the synthetic samples and the baseline sample.
        Return:
        dist_real: distance between the test sample and the baseline sample.
        dist_synthetic: distances between the shuffled samples and the baseline sample.
        """
        # Calculate the rJSD distance between the test sample and the baseline sample.
        dist_real = jensenshannon(self._normalize_cohort(self.test_base_sample[self.subset]),
                                  self._normalize_cohort(self.test_post_ABX_sample[self.subset]))
        # Calculate the rJSD distance between the synthetic samples and the baseline sample.
        dist_synthetic = np.apply_along_axis(lambda x: jensenshannon(
            self._normalize_cohort(self._normalize_cohort(self.test_base_sample[self.subset])), x[self.subset]),
                                             1, self.synthetic_samples)
        return dist_real, dist_synthetic

    def _jaccard(self):
        """
        Calculate the Jaccard distance between the test sample and the baseline sample and the Jaccard distance
        between the synthetic samples and the baseline sample.
        Return:
        dist_real: distance between the test sample and the baseline sample.
        dist_synthetic: distances between the shuffled samples and the baseline sample.
        """
        # Calculate the Jaccard distance between the test sample and the baseline sample.
        dist_real = 1 - Similarity(self.test_base_sample[self.subset], self.test_post_ABX_sample[self.subset],
                                   method="Jaccard").calculate_similarity()
        # Calculate the Jaccard distance between the synthetic samples and the baseline sample.
        dist_synthetic = 1 - np.apply_along_axis(lambda x: Similarity(self.test_base_sample[self.subset],
                                                                      x[self.subset], method="Jaccard"
                                                                      ).calculate_similarity(),
                                                 1, self.synthetic_samples)
        return dist_real, dist_synthetic

    def _weighted_jaccard_symmetric(self):
        """
        Calculate the Symmetric Weighted Jaccard distance between the test sample and the baseline sample and the
        Symmetric Weighted Jaccard distance between the synthetic samples and the baseline sample.
        Return:
        dist_real: distance between the test sample and the baseline sample.
        dist_synthetic: distances between the shuffled samples and the baseline sample.
        """
        # Calculate the Jaccard distance between the test sample and the baseline sample.
        dist_real = 1 - Similarity(self.test_base_sample[self.subset], self.test_post_ABX_sample[self.subset],
                                   method="Weighted Jaccard symmetric").calculate_similarity()
        # Calculate the Jaccard distance between the synthetic samples and the baseline sample.
        dist_synthetic = 1 - np.apply_along_axis(lambda x: Similarity(self.test_base_sample[self.subset],
                                                                      x[self.subset],
                                                                      method="Weighted Jaccard symmetric"
                                                                      ).calculate_similarity(),
                                                 1, self.synthetic_samples)
        return dist_real, dist_synthetic

    def _specificity(self):
        """
        Calculate the Specificity distance between the test sample and the baseline sample and the Specificity distance
        between the synthetic samples and the baseline sample.
        Return:
        dist_real: distance between the test sample and the baseline sample.
        dist_synthetic: distances between the shuffled samples and the baseline sample.
        """
        # Calculate the Specificity distance between the test sample and the baseline sample.
        dist_real = 1 - Similarity(self.test_base_sample[self.subset], self.test_post_ABX_sample[self.subset],
                                   method="Specificity").calculate_similarity()
        # Calculate the Specificity distance between the synthetic samples and the baseline sample.
        dist_synthetic = 1 - np.apply_along_axis(
            lambda x: Similarity(self.test_base_sample[self.subset], x[self.subset],
                                 method="Specificity").calculate_similarity(),
            1, self.synthetic_samples)
        return dist_real, dist_synthetic

    def _recovery(self):
        """
        Calculate the Recovery distance between the test sample and the baseline sample and the Recovery distance
        between the synthetic samples and the baseline sample.
        Return:
        dist_real: distance between the test sample and the baseline sample.
        dist_synthetic: distances between the shuffled samples and the baseline sample.
        """
        # Calculate the Recovery distance between the test sample and the baseline sample.
        dist_real = 1 - Similarity(self.test_base_sample[self.subset], self.test_post_ABX_sample[self.subset],
                                   method="Recovery").calculate_similarity()
        # Calculate the Recovery distance between the synthetic samples and the baseline sample.
        dist_synthetic = 1 - np.apply_along_axis(lambda x: Similarity(self.test_base_sample[self.subset],
                                                                      x[self.subset], method="Recovery"
                                                                      ).calculate_similarity(),
                                                 1, self.synthetic_samples)
        return dist_real, dist_synthetic

    def distance(self, method="Bray Curtis", otu_ids=None, tree=None):
        """
        Inputs:
        method: distance method, from the option: "Bray Curtis", "rJSD", "Jaccard", "Weighted unifrac",
                "Unweighted unifrac", "Specificity", "Recovery".
        otu_ids: list that represent the OTU identifiers.
        tree: phylogenetic tree that corresponds to the OTU identifiers.
        Return:
             distance between the test sample and the baseline sample and the distances between the shuffled samples
             and the baseline sample.
        """
        # Check if the method is valid.
        if method not in ["Bray Curtis", "rJSD", "Jaccard", "Weighted unifrac", "Unweighted unifrac",
                          "Specificity", "Recovery", "Weighted Jaccard symmetric"]:
            raise ValueError("method must be one of the following: Bray Curtis, rJSD, Jaccard,"
                             " Weighted Jaccard symmetric, Weighted unifrac, "
                             "Unweighted unifrac, Recovery, and Specificity.")
        # Calculate the distance between the test sample and the baseline sample for different measures.
        if method == "Bray Curtis":
            return self._bray_curtis()
        elif method == "rJSD":
            return self._rJSD()
        elif method == "Jaccard":
            return self._jaccard()
        elif method == "Weighted Jaccard symmetric":
            self._weighted_jaccard_symmetric()
        elif method == "Weighted unifrac":
            dist_real, dist_synthetic = self._unifrac(method, otu_ids, tree)
            return dist_real, dist_synthetic
        elif method == "Unweighted unifrac":
            dist_real, dist_synthetic = self._unifrac(method, otu_ids, tree)
            return dist_real, dist_synthetic
        elif method == "Specificity":
            return self._specificity()
        elif method == "Recovery":
            return self._recovery()

    def _construct_filtered_data(self, first_sample, second_sample):
        """
        Inputs:
        first_sample: numpy array of shape (# species,).
        second_sample: numpy array of shape (# species,)
        Return:
        filtered_data: normalized numpy matrix of shape (2, # species). Where the ARS species are removed.
        """
        data = np.vstack([first_sample, second_sample])
        filtered_data = self._normalize_cohort(data[:, self.subset].squeeze())
        return filtered_data

    def _construct_pruned_tree(self, otu_ids, tree):
        """
        Inputs:
        otu_ids: List that represent the OTU identifiers.
        tree: The phylogenetic tree that corresponds to the OTU identifiers.
        Returns:
        pruned_tree: The pruned tree that contains only the non-ARS species.
        """
        remove_otu_ids = [otu_ids[i] for i in range(len(otu_ids)) if i in self.subset_comp]
        pruned_tree = tree.copy()
        for node in remove_otu_ids:
            to_delete = pruned_tree.find(node)
            pruned_tree.remove_deleted(lambda x: x == to_delete)
            pruned_tree.prune()
        return pruned_tree

    def _filter_otu_ids(self, otu_ids):
        """
        Inputs:
        otu_ids: List that represent the OTU identifiers.
        Return:
        filtered_otu_ids: List that contains only the non-ARS species.
        """
        return [otu_ids[i] for i in range(len(otu_ids)) if i in self.subset]

    @staticmethod
    def _create_sample_ids(data):
        """
        Inputs:
        data: numpy matrix of shape (2, # species).
        Return:
        sample_ids: List that contains the sample identifiers (the indexes of the samples).
        """
        return np.arange(0, data.shape[0], 1).tolist()

    def _unifrac(self, method, otu_ids=None, tree=None):
        """
        Inputs:
        method: distance method, either "Weighted unifrac" or "Unweighted unifrac".
        otu_ids: list that represent the OTU identifiers.
        tree: phylogenetic tree that corresponds to the OTU identifiers.
        :return:
        dist_real: distance between the test sample and the baseline sample.
        dist_shuffled: distances between the shuffled samples and the baseline sample.
        """
        # Filter the ARS species.
        filtered_data = self._construct_filtered_data(self.test_base_sample, self.test_post_ABX_sample)
        sample_ids = self._create_sample_ids(filtered_data)
        filtered_otu_ids = self._filter_otu_ids(otu_ids)
        pruned_tree = self._construct_pruned_tree(otu_ids, tree)
        # Calculate the distance for different measures.
        if method == "Weighted unifrac":
            dist_real = beta_diversity(metric='weighted_unifrac', counts=filtered_data,
                                       ids=sample_ids, taxa=filtered_otu_ids, tree=pruned_tree, validate=False)[1, 0]
            dist_synthetic = []
            for smp in self.synthetic_samples:
                filtered_data = self._construct_filtered_data(self.test_base_sample, smp)
                dist_synthetic.append(beta_diversity(metric='weighted_unifrac', counts=filtered_data,
                                                    ids=sample_ids, taxa=filtered_otu_ids, tree=pruned_tree,
                                                    validate=False)[1, 0])
            return dist_real, dist_synthetic
        elif method == "Unweighted unifrac":
            dist_real = beta_diversity(metric='unweighted_unifrac', counts=filtered_data,
                                       ids=sample_ids, taxa=filtered_otu_ids, tree=pruned_tree, validate=False)[1, 0]
            dist_synthetic = []
            for smp in self.synthetic_samples:
                filtered_data = self._construct_filtered_data(self.test_base_sample, smp)
                dist_synthetic.append(beta_diversity(metric='unweighted_unifrac', counts=filtered_data,
                                                    ids=sample_ids, taxa=filtered_otu_ids, tree=pruned_tree,
                                                    validate=False)[1, 0])
            return dist_real, dist_synthetic

    def _returned_species(self):
        """
        Find the returned species at each time point.
        Return:
        timepoints_vals: list of boolean numpy arrays of shape (# species,) that represent the returned species at each
                         time point.
        """
        # define the number of time points.
        num_timepoints = self.test_post_ABX_matrix.shape[0]
        # define the operators list.
        op_lst = [operator.ne] * num_timepoints
        # define the general condition.
        general_cond = [self.test_base_sample != 0, self.test_ABX_sample == 0, self.test_post_ABX_sample != 0]
        timepoints_vals = []
        if self.strict:
            # iterate over the time points.
            for j in range(num_timepoints):
                # define the intermediate condition.
                inter_cond = [op_lst[i](self.test_post_ABX_matrix[i, :], 0) for i in range(j + 1)]
                # define the special condition.
                special_cond = [op_lst[i](self.test_post_ABX_matrix[i, :], 0) for i in range(j + 1,
                                                                                             num_timepoints)]
                # combine the conditions.
                cond = general_cond + inter_cond + special_cond
                # find the returned species at each time point.
                timepoints_vals.append(np.logical_and.reduce(cond))
                if j != num_timepoints - 1:
                    # update the operators list.
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