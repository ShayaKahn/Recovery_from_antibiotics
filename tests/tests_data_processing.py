from src.host_specific_recovery.data_processing.functional_data_pipeline import FunctionalDataPipeline
from unittest import TestCase
import numpy as np
import pandas as pd
from pathlib import Path
from src.host_specific_recovery.io.Messaoudene_et_al_loader import (load_Messaoudene_et_al_data,
                                                                    load_Messaoudene_et_al_functional_data)
from src.host_specific_recovery.data_processing.optimal import OptimalCohort
from src.host_specific_recovery.data_processing.rarify import Rarify


class TestOptimalCohort(TestCase):
    def setUp(self) -> None:
        # Two default samples.
        self.samples_dict = {'a': np.array([[11, 0, 8], [3, 9, 2], [0, 1, 3]]),
                             'b': np.array([[7, 1, 2], [1, 6, 0], [2, 3, 8], [8, 2, 5], [0, 1, 0]]),
                             'c': np.array([[35, 0, 17], [3, 4, 3], [1, 0, 8]]),
                             'd': np.array([[12, 7, 4], [1, 0, 0], [7, 1, 0], [6, 6, 6]])}
        self.optimal = OptimalCohort(self.samples_dict, method='jaccard')
        self.optimal_samples = self.optimal.get_optimal_samples()
    def test_get_optimal_samples(self):
        self.assertEqual(np.sum(self.optimal_samples[0], axis=1).tolist(),
                         np.ones(np.size(self.optimal_samples[0], axis=0)).tolist())

class Test_Rarify(TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({
                  'A': [1, 3, 0, 2],
                  'B': [2, 0, 2, 1],
                  'C': [0, 4, 0, 0],
                  'D': [50, 12, 0, 0],
                   })
        print(self.df.sum())
        self.min = Rarify(self.df)
        self.depth = Rarify(self.df, depth=5)

    def test_rarify(self):
        # test the Rarify class using both the default and the depth parameter
        rar_df_min = self.min.rarify()
        self.assertEqual(list(rar_df_min.sum()), [4, 4, 4, 4])
        rar_df_depth = self.depth.rarify()
        self.assertEqual(list(rar_df_depth.sum()), [5, 5, 5])


class TestFunctionalDataPipeline(TestCase):
    """This class tests the FunctionalDataPipeline class."""
    def setUp(self) -> None:

        outputs_data = load_Messaoudene_et_al_data()
        outputs_fun = load_Messaoudene_et_al_functional_data()
        self.rename_map = outputs_fun["rename_map"]
        self.fun_contrib_path = outputs_fun['PATH_contrib_path']

        self.baseline_cols = outputs_data["baseline_df"].columns.tolist()

        self.all_taxa = set(outputs_fun["data"].index.tolist())

        self.N = 10

        self.sample = self.baseline_cols[0]
        self.sample_sur = self.baseline_cols[1]

        def find_baseline_taxa(df, base_cols):
            mask = df[base_cols].gt(0).all(axis=1)
            return set(df.index[mask])

        base_taxa = list(find_baseline_taxa(outputs_fun["data"], [self.sample]))
        self.taxa_dist = list(self.all_taxa - set(base_taxa))

        metadata = pd.read_csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/Metadata.csv")

        self.fun = pd.read_csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/"
                               "DAV132_picrust2_12_12_25_subset/picrust2_out_pipeline_strat/"
                               "EC_metagenome_out/pred_metagenome_unstrat.tsv.gz", sep="\t", index_col=0)

        self.FP = FunctionalDataPipeline(self.fun, self.rename_map, Path(self.fun_contrib_path), outputs_fun["data"])
        self.chunk_size = 500_000

        self.fun_renamed = self.FP.fun_by_sample_table

        self.ASV_table = self.FP.ASV_table

        self.sample_rows_no_agg = self.FP.subsample_stratified_fun(self.sample)

        rng = np.random.default_rng(seed=0)

        self.selected_taxa = rng.choice(self.ASV_table.loc[
                                        self.ASV_table[self.sample] > 0, self.sample].index.to_list(),
                                        size=20, replace=False).tolist()

        self.sample_rows_taxon = self.sample_rows_no_agg["taxon"]
        self.sample_rows_function = self.sample_rows_no_agg["function"]

        self.keep = True

        self.taxa_to_function_set = self.FP.taxa_to_function_sets()

        self.small_ASV_table = pd.DataFrame({
            'base_1': [10, 3, 5, 0, 1, 0, 13],
            'base_2': [2, 15, 0, 3, 15, 0, 14],
            'abx_1': [2, 15, 0, 3, 0, 0, 19],
            'abx_2': [0, 15, 0, 0, 0, 0, 0],
            'post_1': [10, 0, 5, 0, 1, 19, 0],
            'post_2': [10, 16, 5, 3, 1, 11, 0]},
            index=['taxa1', 'taxa2', 'taxa3', 'taxa4', 'taxa5', 'taxa6', 'taxa7'])
