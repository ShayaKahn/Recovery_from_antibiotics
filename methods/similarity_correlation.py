import numpy as np
import pandas as pd
import operator
import random
from methods.similarity import Similarity
from joblib import Parallel, delayed


class SimilarityCorrelation:

    def __init__(self, ABX, baseline, post_ABX_container, baseline_ref, method, timepoints, iters, new=True,
                 strict=True, n_jobs=4):
        """
        ABX: pandas dataframe of shape (# test subjects, # species) that represent the antibiotic treatment samples of
             the tested subjects and the row names are the identifiers of the tested subjects.
        baseline: pandas dataframe of shape (# test subjects, # species) that represent the baseline samples of
                  the tested subjects and the row names are the identifiers of the tested subjects.
        post_ABX_container: dictionary of the post antibiotic treatment samples of the tested subjects. The keys are the
                            identifiers of the tested subjects and the values are numpy matrices of shape
                            (# time points, # species).
        keys_ref: list of strings, the identifiers of the reference subjects (can overlap with the tested subjects keys)
        baseline_ref: pandas dataframe of shape (# ref subjects, # species) that represent the baseline samples of
                      the reference subjects and the row names are the identifiers of the reference subjects.
        method: the method to calculate the similarity correlation. Choose from ' Jaccard' and 'Dice'.
        timepoints: Integer, the number of time points after antibiotics administration the returned species considered
                    as survived.
        iters: Integer, the number of synthetic samples to generate.
        new: Boolean, if True, the correlation is calculated w.r.t the new species, if False, the correlation is
             calculated w.r.t the returned species (at time points >timepoints).
        strict: Boolean, if True, the returned species at time t are the species that are present in the baseline,
                absent in the antibiotic treatment and present in all the post antibiotic samples where t >= timepoints.
                If False, the returned species are the same except that they are present in time t = timepoints and can
                be absent in the post antibiotic samples at time t > timepoints except of the last sample in
                test_post_ABX_matrix.
        n_jobs: Integer, the number of jobs to run in parallel.
        """
        (self.ABX, self.baseline, self.post_ABX_matrices, self.baseline_ref, self.method, self.timepoints, self.iters,
         self.new, self.strict, self.keys, self.keys_ref, self.n_jobs) = self._validate_input(ABX, baseline,
                                                                                              post_ABX_container,
                                                                                              baseline_ref, method,
                                                                                              timepoints,
                                                                                              iters, new, strict,
                                                                                              n_jobs)
        # generate the synthetic cohorts
        self.synthetic_baseline_lst = self._generate_synthetic_cohorts()

    def _validate_input(self, ABX, baseline, post_ABX_container, baseline_ref, method, timepoints, iters, new, strict,
                        n_jobs):
        """
        This method validates the input of the class. The inputs description is in the __init__ function.
        """
        # create list that contains the post_ABX matrices of the test subjects.
        post_ABX_matrices = list(post_ABX_container.values())
        # extract the identifiers of the test subjects.
        keys = list(baseline.columns)
        # extract the identifiers of the reference subjects.
        keys_ref = list(baseline_ref.columns)
        # find the dimensions of all the input samples.
        dims = [post.shape[1] for post in post_ABX_matrices] + [baseline.values.T.shape[1],
                                                                baseline_ref.values.T.shape[1],
                                                                ABX.values.T.shape[1]]
        if not isinstance(ABX, pd.DataFrame):
            raise TypeError("ABX must be a pandas DataFrame")
        if not isinstance(baseline, pd.DataFrame):
            raise TypeError("baseline must be a pandas DataFrame")
        if not isinstance(post_ABX_container, dict):
            raise TypeError("post_ABX_container must be a dictionary")
        if not all(isinstance(post, np.ndarray) for post in post_ABX_matrices):
            raise TypeError("The values of post_ABX_container must be numpy arrays")
        if not isinstance(baseline_ref, pd.DataFrame):
            raise TypeError("baseline_ref must be a pandas DataFrame")
        if method not in ['Jaccard', 'Dice']:
            raise ValueError("Invalid method. Choose from 'Jaccard' and 'Dice'")
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("The keys of the baseline must be strings")
        if not all(isinstance(key, str) for key in keys_ref):
            raise TypeError("The keys of the baseline_ref must be strings")
        if not keys == list(ABX.columns):
            raise ValueError("The keys of the baseline and ABX must be the same")
        if not keys == list(post_ABX_container.keys()):
            raise ValueError("The keys of the baseline and post_ABX_container must be the same")
        if not all(dim == dims[0] for dim in dims):
            raise ValueError("The dimensions are not consistent across samples.")
        if not (isinstance(timepoints, int) and timepoints >= 0):
            raise ValueError("timepoints must be integer greater than or equal to 0")
        if not (isinstance(iters, int) and iters > 0):
            raise ValueError("iters must be integer greater than 0")
        if not isinstance(new, bool):
            raise ValueError("new must be of type bool")
        if not isinstance(strict, bool):
            raise ValueError("strict must be of type bool")
        if not isinstance(n_jobs, int):
            raise ValueError("n_jobs must be of type int")
        return (ABX.values.T, baseline.values.T, post_ABX_matrices, baseline_ref.values.T, method, timepoints, iters,
                new, strict, keys, keys_ref, n_jobs)

    def calc_similarity(self):
        """
        This method generates dictionary. This dictionary contains keys related to the test subjects and the values are
        dictionary that contains the keys related to the reference subjects and the values are tuples of length 2 that
        contain the z-scores that calculated w.r.t the specific reference subject. The first element of the tuple is
        the z-score of the similarity of the new/returned species and the second element is the z-score of the
        similarity of the other species.
        """
        sims_container = {}
        # iterate over the test subjects.
        for base, abx, post_mat, key in zip(self.baseline, self.ABX, self.post_ABX_matrices, self.keys):
            print(key)
            # find the survived and resistant species.
            survived, resistant = self._find_survived_resistant(base, abx, post_mat)
            # find the returned species.
            returned_lst = self._returned_species(base, abx, post_mat)
            # find the indices of the species that are considered in the similarity calculation.
            idx = self._find_subset(base, abx, post_mat, returned_lst)
            # construct the other species set based on the timepoints parameter.
            timpoints_sepcies = [np.where(r)[0] for r in returned_lst[0: self.timepoints]]
            combined_species = [survived, resistant] + timpoints_sepcies
            combined_species = np.hstack(combined_species)
            sims = {}
            # iterate over the reference subjects.
            for j, (base_ref, key_ref) in enumerate(zip(self.baseline_ref, self.keys_ref)):
                if key_ref != key:
                    # calculate the z-scores.
                    sim_new = self._similarity(base_ref[idx], post_mat[-1, :][idx], self.method)
                    sim_new_shuffled = np.array(
                        list(map(lambda x: self._similarity(x[j, :][idx], post_mat[-1, :][idx], self.method),
                                 self.synthetic_baseline_lst)))
                    sim_new_z = self._z_score(sim_new, sim_new_shuffled)
                    sim_others = self._similarity(base_ref[combined_species], post_mat[-1, :][combined_species],
                                                  self.method)
                    sim_others_shuffled = np.array(list(map(lambda x: self._similarity(x[j, :][combined_species],
                                                                                       post_mat[-1, :][
                                                                                           combined_species],
                                                                                       self.method),
                                                            self.synthetic_baseline_lst)))
                    sim_others_z = self._z_score(sim_others, sim_others_shuffled)
                    sims[key_ref] = (sim_new_z, sim_others_z)
            sims_container[key] = sims
        return sims_container

    def _generate_synthetic_cohorts(self):
        """
        This method generates the synthetic cohorts. It returns a list of synthetic cohorts. The length of the list is
        equal to the number of iterations.
        """
        shuffled_baseline_lst = Parallel(n_jobs=self.n_jobs)(
            delayed(self._create_synthetic_cohort)(self.baseline_ref) for _ in range(self.iters))
        return shuffled_baseline_lst

    def _find_subset(self, base, abx, post_mat, returned_lst):
        """
        This method finds the subset of species that are considered in the similarity calculation.
        Inputs:
        base: numpy array of shape (# species,) that represent the baseline sample.
        abx: numpy array of shape (# species,) that represent the antibiotic treatment sample.
        post_mat: numpy array of shape (# time points, # species) that represent the post antibiotic treatment samples.
        returned_lst: list of boolean numpy arrays of shape (# species,) that represent the returned species at each
                      time point.
        Returns: numpy array of shape (# species,) that represent the indices of the species that are considered in the
                 similarity calculation.
        """
        if self.new:
            idx = np.where((base == 0) & (abx == 0) & (post_mat[-1, :] != 0))[0]
        else:
            idx = np.hstack([np.where(r)[0] for r in returned_lst[self.timepoints:]])
        return idx

    @staticmethod
    def _find_survived_resistant(base, abx, post_mat):
        """
        This method finds the survived and resistant species.
        Inputs:
        base: numpy array of shape (# species,) that represent the baseline sample.
        abx: numpy array of shape (# species,) that represent the antibiotic treatment sample.
        post_mat: numpy array of shape (# time points, # species) that represent the post antibiotic treatment samples.
        Returns:
        survived: numpy array of shape (# species,) that represent the survived species.
        resistant: numpy array of shape (# species,) that represent the resistant species.
        """
        survived = np.where((base != 0) & (abx != 0) & (post_mat[-1, :] != 0))[0]
        resistant = np.where((base == 0) & (abx != 0) & (post_mat[-1, :] != 0))[0]
        return survived, resistant

    def _returned_species(self, base, abx, post_mat):
        """
        This method finds the returned species at each time point
        Inputs:
        base: numpy array of shape (# species,) that represent the baseline sample.
        abx: numpy array of shape (# species,) that represent the antibiotic treatment sample.
        post_mat: numpy array of shape (# time points, # species) that represent the post antibiotic treatment samples.
        Returns: list of boolean numpy arrays of shape (# species,) that represent the returned species at each
                 time point.
        """
        # define the number of time points.
        num_timepoints = post_mat.shape[0]
        # initialize the operators list with != operator.
        op_lst = [operator.ne] * num_timepoints
        # define the general condition.
        general_cond = [base != 0, abx == 0, post_mat[-1, :] != 0]
        timepoints_vals = []
        if self.strict:
            # iterate over the time points.
            for j in range(num_timepoints):
                # define the intermediate condition.
                inter_cond = [op_lst[i](post_mat[i, :], 0) for i in range(j + 1)]
                if j != num_timepoints - 1:
                    # define the special condition.
                    special_cond = [op_lst[i](post_mat[i, :], 0) for i in range(j + 1, num_timepoints)]
                    # combine the conditions.
                    cond = general_cond + inter_cond + special_cond
                else:
                    cond = general_cond + inter_cond
                # find the returned species at each time point.
                timepoints_vals.append(np.logical_and.reduce(cond))
                if j != num_timepoints - 1:
                    # update the operators list.
                    op_lst[j] = operator.eq
        else:
            # iterate over the time points.
            for j in range(num_timepoints):
                # define the intermediate condition.
                inter_cond = [op_lst[i](post_mat[i, :], 0) for i in range(j + 1)]
                # combine the conditions.
                cond = general_cond + inter_cond
                # find the returned species at each time point.
                timepoints_vals.append(np.logical_and.reduce(cond))
                if j != num_timepoints - 1:
                    # update the operators list.
                    op_lst[j] = operator.eq
        return timepoints_vals

    @staticmethod
    def _create_synthetic_cohort(cohort):
        """
        This method creates a synthetic cohort by generating samples with the same size (number of nonzero values in
        each sample) of the input reference cohort. The samples are generated baced on the distribution of their
        frequency.
        Input:
        cohort: numpy matrix of shape (# samples, # species) that represent the reference cohort.
        Returns: numpy matrix of shape (# samples, # species) that represent the synthetic cohort.
        """
        # initialize the synthetic cohort.
        synthetic_cohort = np.zeros(cohort.shape)
        # find the pool of species.
        pool = cohort[cohort != 0]
        # find the indices of the species.
        pool_indices = np.where(cohort != 0)[1]
        # iterate over the samples.
        for smp, real in zip(synthetic_cohort, cohort):
            pool_copy = pool.copy()
            pool_indices_copy = pool_indices.copy()
            # find the number of nonzero values in the sample.
            stop = np.size(np.nonzero(real))
            for _ in range(stop):
                # choose a random index.
                index = random.randint(0, len(pool_indices_copy) - 1)
                # update the synthetic sample.
                smp[pool_indices_copy[index]] = pool_copy[index]
                # update the pool.
                mask = pool_indices_copy != pool_indices_copy[index]
                pool_copy = pool_copy[mask]
                pool_indices_copy = pool_indices_copy[mask]
        return synthetic_cohort

    @staticmethod
    def _z_score(p, smp):
        """
        This method calculates the z-score of a given sample.
        Inputs:
        p: float, the value of the sample.
        smp: numpy array of shape (# samples,) that represent the samples.
        Returns: float, the z-score of the sample.
        """
        mu = np.mean(smp)
        sigma = np.std(smp)
        return (p - mu) / sigma

    def _similarity(self, array_first, array_sec, method):
        """
        This method calculates the similarity between two binary samples.
        Inputs:
        array_first: numpy array of shape (# species,) that represent the first sample.
        array_sec: numpy array of shape (# species,) that represent the second sample.
        method: string, the method to calculate the similarity. Choose from 'Jaccard' and 'Dice'.
        Returns: float, the similarity between the two samples.
        """
        return Similarity(array_first, array_sec, method).calculate_similarity()