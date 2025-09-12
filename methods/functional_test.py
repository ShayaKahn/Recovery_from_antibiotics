from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List
from scipy.spatial.distance import braycurtis
from joblib import Parallel, delayed
from pandas.io.parsers.readers import TextFileReader
from typing import Union
import pandas as pd
import csv
import operator

class FunctionalTest:
    """This class is responsible to create the functional profile of a sample from a stratified table."""

    __slots__ = ("sample_name", "sample_table", "reader")

    def __init__(self, stratified_source: Union[pd.DataFrame, str, Path], sample_name: str,
                 chunksize: int = 1_000_000) -> None:
        """Args:
        stratified_source: Path to the stratified table file (can be gzipped) or a pandas DataFrame.
        sample_name: String, name of the sample to extract the functional profile for.
        chunksize: Number of rows to read at a time when reading from file. Default is 1,000,000.
        """

        self.sample_name = str(sample_name)

        # initialize attributes
        self.sample_table = None
        self.reader = None

        # load data
        if isinstance(stratified_source, pd.DataFrame):
            self.sample_table = self._from_dataframe(stratified_source)
        else:
            self.reader = self._from_file(Path(stratified_source), chunksize=chunksize)

    def _from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Load data from dataframe and normalize the functional profile for the specified sample.
        Args:
        df: pandas DataFrame containing the stratified table data.
        Returns:
        A DataFrame with normalized functional profile for the specified sample.
        """

        cols = ["sample", "function", "taxon", "taxon_function_abun"]

        sample_rows = df.loc[df["sample"].eq(self.sample_name), cols].copy()
        if sample_rows.empty:
            raise ValueError(f"Sample '{self.sample_name}' not found")

        # normalize abundances
        total = sample_rows["taxon_function_abun"].sum()

        sample_rows["taxon_function_abun_norm"] = (
                sample_rows["taxon_function_abun"] / total
        )

        return sample_rows

    def _from_file(self, path: Path, chunksize: int = 1_000_000) -> TextFileReader:
        """ Load data from file in chunks.
        Args:
        Path: Path to the stratified table file (can be gzipped).
        chunksize: Number of rows to read at a time. Default is 1,000,000.
        Returns:
        A TextFileReader object that yields DataFrame chunks.
        """

        usecols = ["sample", "function", "taxon", "taxon_function_abun"]

        reader = pd.read_csv(path, sep="\t", compression=("gzip" if path.suffix == ".gz" else None), usecols=usecols,
                             engine="c", chunksize=chunksize, low_memory=False, dtype={"sample": "category",
                                                                                       "function": "category",
                                                                                       "taxon": "category",
                                                                                       "taxon_function_abun": "float32"
                                                                                       },
                             na_filter=False, quoting=csv.QUOTE_NONE)

        for chunk in reader:
            yield chunk

    def _get_sample_species_table_df(self, taxon_list):
        """ Create the functional profile for the specified sample and taxa from DataFrame.
        Args:
        taxon_list: Iterable of taxon names to include in the profile.
        Returns:
        A DataFrame with normalized functional profile for the specified sample and taxa.
        """

        taxa_set = set(taxon_list)
        sub = self.sample_table[self.sample_table["taxon"].isin(taxa_set)]
        return sub[["function", "taxon", "taxon_function_abun_norm"]]

    def _get_sample_species_table_path(self, taxon_list):
        """ Create the functional profile for the specified sample and taxa from file.
        Args:
        taxon_list: Iterable of taxon names to include in the profile.
        Returns:
        A DataFrame with normalized functional profile for the specified sample and taxa.
        """
        sample = self.sample_name
        taxa_list = list(taxon_list)

        frames = []
        sample_total = 0.0

        for chunk in self.reader:

            # filter by sample and taxa
            cs = chunk["sample"]
            ct = chunk["taxon"]

            # get boolean mask for the sample
            sample_code = cs.cat.categories.get_indexer([sample])[0]
            if sample_code == -1:
                continue
            m_sample = (cs.cat.codes.values == sample_code)
            if not np.any(m_sample):
                continue

            # accumulate total abundance for normalization
            vals = chunk["taxon_function_abun"].to_numpy(dtype=np.float32, copy=False)
            sample_total += float(vals[m_sample].sum())

            # get boolean mask for the taxa
            target_codes = ct.cat.categories.get_indexer(taxa_list)
            target_codes = np.asarray(target_codes, dtype=np.int32)
            target_codes = target_codes[target_codes >= 0]
            if target_codes.size == 0:
                continue

            # get boolean mask for the taxa present in the chunk
            m_taxa = np.isin(ct.cat.codes.values, target_codes, assume_unique=False)
            m = m_sample & m_taxa
            if not m.any():
                continue

            # gather needed columns only
            idx = np.flatnonzero(m)
            frames.append(pd.DataFrame({
                "function": chunk["function"].iloc[idx].astype("string").to_numpy(),
                "taxon": chunk["taxon"].iloc[idx].astype("string").to_numpy(),
                "taxon_function_abun": vals[idx],
            }))

            if not frames:
                raise ValueError(f"No rows found for sample='{self.sample_name}' "f"and the supplied taxon list.")

            sub = pd.concat(frames, ignore_index=True)
            sub["taxon_function_abun_norm"] = (sub["taxon_function_abun"] / sample_total)

            return sub[["function", "taxon", "taxon_function_abun_norm"]]

    def get_sample_species_table(self, taxon_list):
        if self.sample_table is not None:
            return self._get_sample_species_table_df(taxon_list)
        elif self.reader is not None:
            return self._get_sample_species_table_path(taxon_list)
        else:
            raise ValueError("No data available to create sample species table.")

class ApplyFunctionalTest:
    """This class applies the functional test to a dataset."""

    __slots__ = ("base_col", "abx_col", "post_cols", "data", "new", "dis", "size_new", "size_dis", "size_post",
                 "size_base", "fun_new", "fun_dis", "fun_new_obj", "fun_dis_obj", "stratified_source", "name_to_run",
                 "unstratified_source", "fun_base_norm", "fun_ABX_norm", "fun_post_norm", "fun_new_grouped",
                 "fun_dis_grouped", "iters", "new_null_cont", "fun_new_null_cont", "fun_new_null_grouped_cont",
                 "verbose", "n_jobs", "chunksize", "sur_cols", "sur_cols_cont", "new_type", "fun_new_grouped_pad",
                 "fun_dis_grouped_pad", "fun_new_grouped_pad_norm", "fun_dis_grouped_pad_norm")

    def __init__(self, unstratified_source: pd.DataFrame, stratified_source: Union[Path, pd.DataFrame],
                 name_to_run: dict[str, str], base_col: str, abx_col: str, post_cols: List[str], sur_cols: List[str],
                 data: pd.DataFrame, iters: int, verbose: bool = True, n_jobs: Union[int, None] = -1,
                 chunksize: int = 1_000_000, new_type: str = "simple") -> None:
        """Args:
        unstratified_source: DataFrame containing the unstratified table data.
        stratified_source: Path to the stratified table file (can be gzipped) or a pandas DataFrame.
        name_to_run: Dictionary mapping column names to sample names in the stratified table.
        base_col: String, name of the baseline column in the data DataFrame.
        abx_col: String, name of the antibiotic treatment column in the data DataFrame.
        post_cols: List of strings, names of the post-treatment columns in the data DataFrame.
        sur_cols: List of strings, names of the surrogate columns in the data DataFrame.
        data: DataFrame containing the species abundance data.
        iters: Number of iterations for generating null distributions.
        verbose: Boolean, whether to print progress messages. Default is True.
        n_jobs: Number of parallel jobs to run. Default is -1 (use all available cores).
        chunksize: Number of rows to read at a time when reading from file. Default is 1,000,000.
        new_type: String, type of 'new' species definition ("simple", "soft", or "strict"). Default is "simple".
        """

        self.base_col = base_col
        self.abx_col = abx_col
        self.post_cols = post_cols
        self.sur_cols = sur_cols
        self.data = data
        self.stratified_source = stratified_source
        self.name_to_run = name_to_run
        self.unstratified_source = unstratified_source
        self.iters = iters
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.new_type = new_type
        self.new = self._find_new()
        self.size_new = len(self.new)
        self.dis = self._find_dis()
        self.size_dis = len(self.dis)
        self.size_post = (self.data[self.post_cols[-1]] > 0).sum()
        self.size_base = (self.data[self.base_col] > 0).sum()
        self.fun_new_obj = FunctionalTest(self.stratified_source, self.name_to_run[self.post_cols[-1]], self.chunksize)
        self.fun_dis_obj = FunctionalTest(self.stratified_source, self.name_to_run[self.base_col], self.chunksize)
        self.fun_new = self.fun_new_obj.get_sample_species_table(self.new)
        self.fun_dis = self.fun_dis_obj.get_sample_species_table(self.dis)
        self.fun_new_grouped, self.fun_dis_grouped = self._find_grouped()
        self.fun_new_grouped_pad, self.fun_dis_grouped_pad = self._pad_fun_grouped(self.fun_new_grouped,
                                                                                   self.fun_dis_grouped)
        self.fun_new_grouped_pad_norm, self.fun_dis_grouped_pad_norm = self._normalize_pad_fun_grouped(
            self.fun_new_grouped_pad, self.fun_dis_grouped_pad)
        self.new_null_cont, self.sur_cols_cont = self._find_new_null_iters()
        self.fun_new_null_cont, self.fun_new_null_grouped_cont = self._find_new_null_fun_opt()

    def _find_new_simple(self):
        """ A species is considered 'new' using the 'simple' definition if it is absent in both the baseline and
         antibiotic treatment"""
        new = (self.data[self.post_cols[-1]] > 0) & (self.data[self.base_col] == 0) & (self.data[self.abx_col] == 0)
        return new[new].index.tolist()

    def _find_new_soft(self):
        """
        A species is considered 'new' using the 'soft' definition if it is absent in both the baseline and
        antibiotic treatment, and appears at least once before the final post-ABX state
        in the post-treatment time points.
        Return:
        A list of indices of the 'new' species.
        """
        num_timepoints = len(self.post_cols)
        op_lst = [operator.ne] * num_timepoints
        general_cond = [(self.data[self.base_col] == 0), (self.data[self.abx_col] == 0),
                        (self.data[self.post_cols[-1]] > 0)]
        timepoints_vals = []
        for j in range(num_timepoints):
            # define the intermediate condition.
            inter_cond = [op_lst[i](self.data[self.post_cols[i]], 0) for i in range(j + 1)]
            # combine the conditions.
            cond = general_cond + inter_cond
            # find the returned species at each time point.
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                # update the operators list.
                op_lst[j] = operator.eq
        new = np.logical_or.reduce(timepoints_vals[:-1])
        return list(self.data.index[new])

    def _find_new_strict(self):
        """
        A species is considered 'new' using the 'strict' definition if it is absent in both the baseline and
        antibiotic treatment, appears at least once before the final post-ABX state in the post-treatment time points,
        and once it appears it remains present in all subsequent post-treatment time points.
        Return:
        A list of indices of the 'new' species.
        """
        num_timepoints = len(self.post_cols)
        op_lst = [operator.ne] * num_timepoints
        general_cond = [(self.data[self.base_col] == 0), (self.data[self.abx_col] == 0),
                        (self.data[self.post_cols[-1]] > 0)]
        timepoints_vals = []
        for j in range(num_timepoints):
            inter_cond = [op_lst[i](self.data[self.post_cols[i]], 0) for i in range(j + 1)]
            special_cond = [op_lst[i](self.data[self.post_cols[i]],
                                      0) for i in range(j + 1, num_timepoints)]
            cond = general_cond + inter_cond + special_cond
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                op_lst[j] = operator.eq

        new = np.logical_or.reduce(timepoints_vals[:-1])
        return list(self.data.index[new])

    def _find_new(self):
        """
        Determine the 'new' species based on the specified new_type.
        Return:
        A list of indices of the 'new' species.
        """
        if self.new_type == "simple":
            return self._find_new_simple()
        elif self.new_type == "soft":
            return self._find_new_soft()
        elif self.new_type == "strict":
            return self._find_new_strict()
        else:
            raise ValueError(f"Unknown new type: {self.new_type}")

    def _find_dis(self):
        """
        A species is considered 'disappeared' if it is present in the baseline, absent in the antibiotic treatment,
        and absent in all post-treatment time points.
        Return:
        A list of indices of the 'disappeared' species.
        """
        dis = ((self.data[self.post_cols].sum(axis=1) == 0) & (
                self.data[self.base_col] > 0) & (self.data[self.abx_col] == 0))
        return dis[dis].index.tolist()

    def _find_grouped(self):
        """
        Group the functional profiles by function and sum the normalized abundances.
        Return:
        Two Series objects containing the grouped functional profiles for 'new' and 'disappeared' species.
        """
        fun_new_grouped = self.fun_new.groupby("function", as_index=True)["taxon_function_abun_norm"].agg("sum")
        fun_dis_grouped = self.fun_dis.groupby("function", as_index=True)["taxon_function_abun_norm"].agg("sum")
        return fun_new_grouped, fun_dis_grouped

    def _find_new_null(self, sur_col):
        """
        Generate a new null set of species for the given surrogate column.
        sur_col: String, name of the surrogate column in the data DataFrame.
        Return:
        A list of indices of the new null set of species, or None if not enough candidates.
        """
        new_null_cond = (self.data[sur_col] > 0)
        new_null_pool = new_null_cond[new_null_cond].index.tolist()
        new_null_pool = [x for x in new_null_pool if x not in self.dis]
        if len(new_null_pool) <= len(self.new):
            new_null = None
        else:
            new_null = list(np.random.choice(new_null_pool, size=len(self.new), replace=False))
        return new_null

    def _find_new_null_iters(self):
        """
        Generate multiple new null sets of species for each surrogate column.
        Return:
        Two lists: one containing the new null sets and another with the corresponding surrogate columns.
        """
        new_null_cont = []
        sur_cols_cont = []
        for i, sur_col in enumerate(self.sur_cols):
            if self.verbose:
                print(f"Running iteration {i + 1} for the creation of new null sets.")
            new_null = self._find_new_null(sur_col)
            if new_null is None:
                continue
            else:
                new_null_cont.append(new_null)
                sur_cols_cont.append(sur_col)
            for j in range(self.iters - 1):
                new_null = self._find_new_null(sur_col)
                new_null_cont.append(new_null)
                sur_cols_cont.append(sur_col)
        return new_null_cont, sur_cols_cont

    def _find_new_null_fun(self):
        """
        Create the functional profiles for each new null set of species.
        Return:
        Two lists: one containing the functional profiles for each new null set and another with the grouped profiles.
        """
        fun_new_null_cont = []
        fun_new_null_grouped_cont = []
        for j, (new_null, sur_col) in enumerate(zip(self.new_null_cont, self.sur_cols_cont)):
            if self.verbose:
                print(f"Running iteration number {j+1} for the creation of the functional profiles of new null sets.")
            fun_new_null_obj = FunctionalTest(self.stratified_source, self.name_to_run[sur_col], self.chunksize)
            fun_new_null = fun_new_null_obj.get_sample_species_table(new_null)
            fun_new_null_grouped = fun_new_null.groupby("function", as_index=True)["taxon_function_abun_norm"].agg("sum")
            fun_new_null_cont.append(fun_new_null)
            fun_new_null_grouped_cont.append(fun_new_null_grouped)
        return fun_new_null_cont, fun_new_null_grouped_cont

    def _find_new_null_fun_for_j(self, j, new_null, sur_col):
        """
        Create the functional profile for a specific new null set of species.
        Aegs:
        j: Index of the current iteration.
        new_null: List of indices of the new null set of species.
        sur_col: String, name of the surrogate column in the data DataFrame.
        Return:
        A tuple containing the functional profile DataFrame and the grouped profile Series for the new null set.
        """
        if self.verbose:
            print(f"Running iteration number {j+1} for the creation of the functional profiles of new null sets.\n")
        fun_new_null_obj = FunctionalTest(self.stratified_source, self.name_to_run[sur_col], self.chunksize)
        fun_new_null = fun_new_null_obj.get_sample_species_table(new_null)
        fun_new_null_grouped = fun_new_null.groupby("function", as_index=True)["taxon_function_abun_norm"].agg("sum")
        return fun_new_null, fun_new_null_grouped

    def _run_parallel_new_null_fun(self):
        """
        Create the functional profiles for each new null set of species in parallel.
        Return:
        Two lists: one containing the functional profiles for each new null set and another with the grouped profiles.
        """
        if self.verbose:
            print("Running parallel computation for the functional profiles of new null sets.")
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._find_new_null_fun_for_j)(j, new_null, sur_col) for j, (new_null, sur_col) in enumerate(
                zip(self.new_null_cont, self.sur_cols_cont))
        )
        fun_new_null_cont, fun_new_null_grouped_cont = zip(*results)
        return list(fun_new_null_cont), list(fun_new_null_grouped_cont)

    def _find_new_null_fun_opt(self):
        """
        Create the functional profiles for each new null set of species, either sequentially or in parallel.
        Return:
        Two lists: one containing the functional profiles for each new null set and another with the grouped profiles.
        """
        if (self.n_jobs is None) or (self.n_jobs == 1):
            return self._find_new_null_fun()
        else:
            return self._run_parallel_new_null_fun()

    @staticmethod
    def _pad_fun_grouped(fun_new_grouped, fun_dis_grouped):
        """
        Pad the grouped functional profiles to have the same index.
        fun_new_grouped: Series containing the grouped functional profile for 'new' species.
        fun_dis_grouped: Series containing the grouped functional profile for 'disappeared' species.
        Return:
        Two numpy arrays containing the padded functional profiles for 'new' and 'disappeared' species.
        """
        idx_new_dis_union = fun_new_grouped.index.union(fun_dis_grouped.index)
        fun_new_grouped_pad = fun_new_grouped.reindex(idx_new_dis_union, fill_value=0.).astype(float).to_numpy()
        fun_dis_grouped_pad = fun_dis_grouped.reindex(idx_new_dis_union, fill_value=0.).astype(float).to_numpy()
        return fun_new_grouped_pad, fun_dis_grouped_pad

    @staticmethod
    def _normalize_pad_fun_grouped(fun_new_grouped_pad, fun_dis_grouped_pad):
        """
        Normalize the padded functional profiles.
        fun_new_grouped_pad: Numpy array containing the padded functional profile for 'new' species.
        fun_dis_grouped_pad: Numpy array containing the padded functional profile for 'disappeared' species.
        Return:
        Two numpy arrays containing the normalized functional profiles for 'new' and 'disappeared' species.
        """
        fun_new_grouped_pad_norm = fun_new_grouped_pad / fun_new_grouped_pad.sum(axis=0, keepdims=True)
        fun_dis_grouped_pad_norm = fun_dis_grouped_pad / fun_dis_grouped_pad.sum(axis=0, keepdims=True)
        return fun_new_grouped_pad_norm, fun_dis_grouped_pad_norm

    def _calc_null_dist_for_j(self, j, fun_new_null_grouped):
        """
        Calculate the Bray-Curtis similarity for a specific new null set of species.
        j: Index of the current iteration.
        fun_new_null_grouped: Series containing the grouped functional profile for the new null set of species.
        Return:
        A float representing the Bray-Curtis similarity for the new null set.
        """
        if self.verbose:
            print(f"Running iteration number {j + 1} for the functional test of the new null sets.")
        fun_new_null_grouped_pad, fun_dis_null_grouped_pad = self._pad_fun_grouped(fun_new_null_grouped,
                                                                                   self.fun_dis_grouped)
        fun_new_null_grouped_pad_norm, fun_dis_null_grouped_pad_norm = self._normalize_pad_fun_grouped(
            fun_new_null_grouped_pad, fun_dis_null_grouped_pad)
        null_sim = 1. - braycurtis(fun_new_null_grouped_pad_norm, fun_dis_null_grouped_pad_norm)
        return null_sim

    def _calc_null_sims_parallel(self):
        """
        Calculate the Bray-Curtis similarities for all new null sets of species in parallel.
        Return:
        A list of floats representing the Bray-Curtis similarities for each new null set.
        """
        if self.verbose:
            print("Running parallel computation for the distance calculations for the new null sets.")
        null_sims = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._calc_null_dist_for_j)(
            j, fun_new_null_grouped) for j, fun_new_null_grouped in enumerate(self.fun_new_null_grouped_cont))
        return list(null_sims)

    def get_results(self):
        """
        Calculate the real similarity and null similarities, and compute the p-value.
        Return:
        A dictionary containing the real similarity, null similarities, p-value, and sizes of the species sets.
        """
        real_sim = 1. - braycurtis(self.fun_new_grouped_pad_norm, self.fun_dis_grouped_pad_norm)
        if (self.n_jobs is None) or (self.n_jobs == 1):
            null_sims = [0.] * len(self.fun_new_null_grouped_cont)
            for j, fun_new_null_grouped in enumerate(self.fun_new_null_grouped_cont):
                if self.verbose:
                    print(f"Running iteration number {j+1} for the functional test of the new null sets.")
                fun_new_null_grouped_pad, fun_dis_null_grouped_pad = self._pad_fun_grouped(fun_new_null_grouped,
                                                                                           self.fun_dis_grouped)
                fun_new_null_grouped_pad_norm, fun_dis_null_grouped_pad_norm = self._normalize_pad_fun_grouped(
                    fun_new_null_grouped_pad, fun_dis_null_grouped_pad)
                null_sim = 1. - braycurtis(fun_new_null_grouped_pad_norm, fun_dis_null_grouped_pad_norm)
                null_sims[j] = null_sim
        else:
            null_sims = self._calc_null_sims_parallel()
        p_val = np.mean(np.array(null_sims) >= real_sim)
        results = {
            "real_similarity": real_sim,
            "null_similarities": null_sims,
            "p_value": p_val,
            "size_new": self.size_new,
            "size_dis": self.size_dis,
            "size_post": self.size_post,
            "size_base": self.size_base,
        }
        return results
