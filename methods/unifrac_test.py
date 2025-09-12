from skbio.diversity import beta_diversity
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import operator

class UnifracTest:
    """This class implements the Unifrac test."""

    __slots__ = ('tree', 'feature_table', 'test_ids', 'pattern', 'days', 'add_info', 'filtered_biom',
                 'shared_otus', 'filtered_feature_table', 'num_last_days', 'surrogate_iters', 'n_jobs', 'verbose',
                 'new_type')
    def __init__(self, feature_table, tree, test_ids, pattern, days, add_info, num_last_days = 1, surrogate_iters = 1,
                 n_jobs=-1, verbose=True, new_type='simple'):
        """Initialize the UnifracTest class.
        Args:
            feature_table (Pandas DataFrame): Feature table with shape (n_taxa, n_samples).
            tree: Phylogenetic tree represented as instance of the class skbio.tree._tree.TreeNode.
            test_ids (list): List of subject IDs to be used for testing.
            pattern (str): Pattern that separates subject ID from the sample name in the feature table.
            days (list): List contains strings that represent the days in the data.
            add_info (list: Additional information to the name of the sample for each subject.
            num_last_days (int): Number of last days to consider for the surrogate calculations.
            surrogate_iters (int): Number of iterations for a surrogate sample calculations.
            n_jobs (int): Number of jobs to run in parallel. Default is -1, which uses all available cores."""

        self.feature_table = feature_table
        self.tree = tree
        self.test_ids = test_ids
        self.pattern = pattern
        self.days = days
        self.add_info = add_info
        self.num_last_days = num_last_days
        self.surrogate_iters = surrogate_iters
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.new_type = new_type
        self.shared_otus = self._find_shared_otus()
        self.filtered_feature_table = self.feature_table.loc[self.shared_otus]

    def _find_shared_otus(self):
        """Find shared OTUs between the feature table and the phylogenetic tree."""

        tree_tips = list(set(tip.name for tip in self.tree.tips()))
        otu_ids = list(self.feature_table.index)
        shared_otus = [o for o in otu_ids if o in tree_tips]
        return shared_otus

    def _find_new_simple(self, base_col, abx_col, post_col):
        """Find new taxa in the post-antibiotic sample.
        Args:
            base_col (str): Column name for the baseline sample.
            abx_col (str): Column name for the antibiotic sample.
            post_col (str): Column name for the post-antibiotic sample.
        Returns:
            new (np.ndarray): Boolean array indicating new taxa."""

        new = (self.filtered_feature_table[post_col] > 0) & (self.filtered_feature_table[base_col] == 0) & (
               self.filtered_feature_table[abx_col] == 0)
        return new.values

    def _find_new_soft(self, base_col, abx_col, post_cols):
        """
        Find new taxa in the post-antibiotic samples using a soft criterion.
        Args:
            base_col (str): Column name for the baseline sample.
            abx_col (str): Column name for the antibiotic sample.
            post_cols (list): List of column names for the post-antibiotic samples.
        Returns:
            new (np.ndarray): Boolean array indicating new taxa.
        """
        num_timepoints = self.filtered_feature_table[post_cols].shape[1]
        op_lst = [operator.ne] * num_timepoints
        general_cond = [(self.filtered_feature_table[base_col] == 0), (self.filtered_feature_table[abx_col] == 0),
                        (self.filtered_feature_table[post_cols[-1]] > 0)]
        timepoints_vals = []
        for j in range(num_timepoints):
            # define the intermediate condition.
            inter_cond = [op_lst[i](self.filtered_feature_table[post_cols[i]], 0) for i in range(j + 1)]
            # combine the conditions.
            cond = general_cond + inter_cond
            # find the returned species at each time point.
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                # update the operators list.
                op_lst[j] = operator.eq
        return np.logical_or.reduce(timepoints_vals[:-1])

    def _find_new_strict(self, base_col, abx_col, post_cols):
        """
        Find new taxa in the post-antibiotic samples using a strict criterion.
        Args:
            base_col (str): Column name for the baseline sample.
            abx_col (str): Column name for the antibiotic sample.
            post_cols (list): List of column names for the post-antibiotic samples.
        Returns:
            new (np.ndarray): Boolean array indicating new taxa.
        """
        num_timepoints = self.filtered_feature_table[post_cols].shape[1]
        op_lst = [operator.ne] * num_timepoints
        general_cond = [(self.filtered_feature_table[base_col] == 0), (self.filtered_feature_table[abx_col] == 0),
                        (self.filtered_feature_table[post_cols[-1]] > 0)]
        timepoints_vals = []
        for j in range(num_timepoints):
            inter_cond = [op_lst[i](self.filtered_feature_table[post_cols[i]], 0) for i in range(j + 1)]
            special_cond = [op_lst[i](self.filtered_feature_table[post_cols[i]],
                                      0) for i in range(j + 1, num_timepoints)]
            cond = general_cond + inter_cond + special_cond
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                op_lst[j] = operator.eq
        return np.logical_or.reduce(timepoints_vals[:-1])

    def _find_new(self, base_col, abx_col, post_cols):
        """Find new taxa in the post-antibiotic samples based on the specified new_type.
        Args:
            base_col (str): Column name for the baseline sample.
            abx_col (str): Column name for the antibiotic sample.
            post_cols (list): List of column names for the post-antibiotic samples.
        Returns:
            new (np.ndarray): Boolean array indicating new taxa.
        """
        if self.new_type == 'simple':
            return self._find_new_simple(base_col, abx_col, post_cols[-1])
        elif self.new_type == 'soft':
            return self._find_new_soft(base_col, abx_col, post_cols)
        elif self.new_type == 'strict':
            return self._find_new_strict(base_col, abx_col, post_cols)
        else:
            raise ValueError(f"Unknown new_type: {self.new_type}. Choose from 'simple', 'soft', or 'strict'.")

    def _find_dis(self, base_col, abx_col, post_cols):
        """Find taxa that disappeared.
        Args:
            base_col (str): Column name for the baseline sample.
            abx_col (str): Column name for the antibiotic sample.
            post_cols (list): List of column names for the post-antibiotic samples.
        Returns:
            dis (np.ndarray): Boolean array indicating disappeared taxa."""

        dis = ((self.filtered_feature_table[post_cols].sum(axis=1) == 0) & (
                self.filtered_feature_table[base_col] > 0) & (self.filtered_feature_table[abx_col] == 0))
        return dis.values

    def _unifrac_two_samples(self, new, dis):
        """Calculate the unweighted UniFrac distance between two samples.
        Args:
            new (np.ndarray): Boolean array indicating new taxa.
            dis (np.ndarray): Boolean array indicating disappeared taxa.
        Returns:
            float: Unweighted UniFrac distance between the two samples."""

        otu = np.zeros((2, self.filtered_feature_table.shape[0]), dtype=int)
        otu[0, dis] = 1
        otu[1, new] = 1

        sample_ids = ['dis', 'new']

        otu_df = pd.DataFrame(otu, index=sample_ids, columns=self.shared_otus)

        return beta_diversity(metric="unweighted_unifrac", counts=otu_df,
                              ids=sample_ids, tree=self.tree,
                              taxa=self.shared_otus, validate=True).data[0, 1]

    def _find_pool(self, dis, surrogate_col):
        """Find the pool of taxa that are present in the surrogate column but not in the disappeared taxa of the test
        subject.
        Args:
            dis (np.ndarray): Boolean array indicating disappeared taxa.
            surrogate_col (str): Column name for the surrogate sample.
        Returns:
            pool (np.ndarray): Indices of taxa that are present in the surrogate column but not in the disappeared taxa."""

        dis_idx = np.nonzero(dis)[0]
        present_taxa = (self.filtered_feature_table[surrogate_col] != 0)
        present_taxa_idx = np.nonzero(present_taxa)[0]
        pool = present_taxa_idx[~np.isin(present_taxa_idx, dis_idx)]
        return pool

    def _find_new_null(self, new, surrogate_col, pool):
        """Find a null new species for the surrogate column.
        Args:
            new (np.ndarray): Boolean array indicating new taxa.
            surrogate_col (str): Column name for the surrogate sample.
            pool (np.ndarray): Indices of taxa that are present in the surrogate column but not in the disappeared taxa.
        Returns:
            new_null (np.ndarray): Boolean array indicating null new taxa."""

        n = new.sum()
        n_eff = n if n < np.size(pool) else np.size(pool)
        new_null_idx = np.random.choice(pool, size=n_eff, replace=False)
        new_null = np.zeros(len(self.filtered_feature_table[surrogate_col]), dtype=bool)
        new_null[new_null_idx] = True
        return new_null

    def apply_test(self):
        """Apply the Unifrac test to the feature table.
        Returns:
            results_real (list): List of real UniFrac distances for each subject.
            results_surrogate_container (list): List of surrogate UniFrac distances for each subject."""

        results_surrogate_container = []
        results_real = [0.] * len(self.test_ids)

        surrogate_clos = []
        for i in range(1, self.num_last_days + 1):
            surrogate_clos += [col for col in self.feature_table.columns if f"{self.pattern}{self.days[-i]}" in col]

        results = Parallel(n_jobs=self.n_jobs)(delayed(self._apply_test_for_subject)(
            i, surrogate_clos) for i in range(len(self.test_ids)))

        for i, (real, surrogate) in enumerate(results):
            results_real[i] = real
            results_surrogate_container.append(surrogate)
        return results_real, results_surrogate_container

    def _apply_test_for_subject(self, i, surrogate_clos):
        """Apply the Unifrac test for a specific subject.
        Args:
            i (int): Index of the subject in the test_ids list.
            surrogate_clos (list): List of surrogate columns.
        Returns:
            tuple: Real UniFrac distance and list of surrogate UniFrac distances for the subject.
        """

        if self.verbose:
            print(f"Processing subject {self.test_ids[i]}...")
        post_cols = [f"{self.test_ids[i]}{self.pattern}{day}{self.add_info[i]}" for day in self.days[2:]]
        # Find new taxa and disappeared taxa for the subject
        new = self._find_new(f"{self.test_ids[i]}{self.pattern}{self.days[0]}{self.add_info[i]}",
                             f"{self.test_ids[i]}{self.pattern}{self.days[1]}{self.add_info[i]}",
                             post_cols)#f"{self.test_ids[i]}{self.pattern}{self.days[-1]}{self.add_info[i]}")
        dis = self._find_dis(f"{self.test_ids[i]}{self.pattern}{self.days[0]}{self.add_info[i]}",
                             f"{self.test_ids[i]}{self.pattern}{self.days[1]}{self.add_info[i]}",
                             post_cols)
        if self.verbose:
            print(f"New taxa for subject {self.test_ids[i]}: {new.sum()}")
            print(f"Disappeared taxa for subject {self.test_ids[i]}: {dis.sum()}")
        # Calculate the real UniFrac distance
        real = 1. - self._unifrac_two_samples(new, dis)

        # Calculate surrogate UniFrac distances
        results_surrogate = []
        for surrogate_col in surrogate_clos:
            if f"{self.test_ids[i]}{self.pattern}" not in surrogate_col:
                pool = self._find_pool(dis, surrogate_col)
                if np.size(pool) > new.sum():
                    for _ in range(self.surrogate_iters):
                        new_null = self._find_new_null(new, surrogate_col, pool)
                        results_surrogate.append(1. - self._unifrac_two_samples(new_null, dis))
        return real, results_surrogate
