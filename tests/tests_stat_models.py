from src.host_specific_recovery.statistical_models.null_model import NullModel
from src.host_specific_recovery.statistical_models.surrogate import Surrogate
from src.host_specific_recovery.statistical_models.similarity_correlation import SimilarityCorrelation
from src.host_specific_recovery.statistical_models.functional_test import FunctionalTest, ApplyFunctionalTest
from src.host_specific_recovery.statistical_models.unifrac_test import UnifracTest
from src.host_specific_recovery.statistical_models.phylogenetic_test import PhylogeneticTest
from src.host_specific_recovery.analysis.phylogenetic_analysis import run_phylogenetic_test
from unittest import TestCase
import numpy as np
import pandas as pd
from pandas.io.parsers.readers import TextFileReader
from pathlib import Path
from skbio import TreeNode


class TestPhylogeneticTest(TestCase):
    def setUp(self) -> None:
        taxa = [
            "lost",
            "other_baseline",
            "transient",
            "last_only",
            "colonizer",
            "early_and_colonizer",
            "baseline_partial",
            "abx_present",
        ]
        self.base = pd.DataFrame(
            {
                "base_1": [1, 1, 0, 0, 0, 0, 1, 0],
                "base_2": [1, 1, 0, 0, 0, 0, 0, 0],
            },
            index=taxa,
        )
        self.abx = pd.Series(
            [0, 0, 0, 0, 0, 0, 0, 1],
            index=taxa,
            name="abx_1",
        )
        self.post = pd.DataFrame(
            {
                "post_1": [0, 1, 0, 0, 1, 1, 0, 1],
                "post_2": [0, 0, 1, 0, 1, 1, 0, 0],
                "post_3": [0, 1, 0, 1, 1, 1, 1, 0],
            },
            index=taxa,
        )
        self.tree = (
            "((lost:1,other_baseline:1):1,"
            "(transient:1,(colonizer:1,early_and_colonizer:1):1):1)root;"
        )
        self.test = PhylogeneticTest(self.base, self.abx, self.post, self.tree)

    def test_find_lost_species(self):
        self.assertEqual(self.test.find_lost_species(), {"lost"})

    def test_find_other_baseline(self):
        self.assertEqual(self.test.find_other_baseline(), {"other_baseline"})

    def test_find_transient(self):
        self.assertEqual(self.test.find_transient(), {"transient"})

    def test_find_colonizers(self):
        self.assertEqual(self.test.find_colonizers(), {"colonizer", "early_and_colonizer"})

    def test_find_species_categories(self):
        self.assertEqual(
            self.test.find_species_categories(),
            {
                "lost_species": {"lost"},
                "other_baseline": {"other_baseline"},
                "transient": {"transient"},
                "colonizers": {"colonizer", "early_and_colonizer"},
            },
        )

    def test_compute_unifrac_similarities(self):
        similarities = self.test.compute_unifrac_similarities()

        self.assertEqual(
            set(similarities),
            {
                "colonizers_lost",
                "colonizers_other_baseline",
                "transient_lost",
                "transient_other_baseline",
                "lost_species_other_baseline",
                "colonizers_transient",
            },
        )
        self.assertEqual(similarities["colonizers_lost"], 0.0)
        self.assertEqual(similarities["colonizers_other_baseline"], 0.0)
        self.assertEqual(similarities["transient_lost"], 0.0)
        self.assertEqual(similarities["transient_other_baseline"], 0.0)
        self.assertAlmostEqual(similarities["lost_species_other_baseline"], 1 / 3)
        self.assertAlmostEqual(similarities["colonizers_transient"], 0.2)

    def test_unifrac_similarity_preserves_underscores_in_newick_taxa(self):
        self.assertEqual(
            self.test._unifrac_similarity_between_taxa_sets({"other_baseline"}, {"other_baseline"}),
            1.0,
        )

    def test_accepts_single_baseline_sample(self):
        one_baseline_sample = self.base["base_1"]
        test = PhylogeneticTest(one_baseline_sample, self.abx, self.post, self.tree)

        self.assertEqual(test.find_lost_species(), {"lost"})

    def test_rejects_multiple_abx_samples(self):
        abx_samples = pd.DataFrame(
            {
                "abx_1": self.abx,
                "abx_2": self.abx,
            }
        )

        with self.assertRaises(ValueError):
            PhylogeneticTest(self.base, abx_samples, self.post, self.tree)


class TestPhylogeneticAnalysis(TestCase):
    def setUp(self) -> None:
        self.taxa = [
            "lost",
            "other_baseline",
            "transient",
            "last_only",
            "colonizer",
            "early_and_colonizer",
            "baseline_partial",
            "abx_present",
        ]
        self.dataset = {
            "filtered_keys": ["subject_a", "subject_b"],
            "baseline_df": pd.DataFrame(
                {
                    "a_base": [1, 1, 0, 0, 0, 0, 0, 0],
                    "b_base": [1, 1, 0, 0, 0, 0, 0, 0],
                },
                index=self.taxa,
            ),
            "baseline_full_df": pd.DataFrame(
                {
                    "full_1": [0, 0, 1, 1, 1, 1, 1, 1],
                    "full_2": [0, 0, 1, 1, 1, 1, 1, 1],
                },
                index=self.taxa,
            ),
            "abx_df": pd.DataFrame(
                {
                    "a_abx": [0, 0, 0, 0, 0, 0, 0, 1],
                    "b_abx": [0, 0, 0, 0, 0, 0, 0, 1],
                },
                index=self.taxa,
            ),
            "post_abx_cohorts_df": [
                pd.DataFrame(
                    {
                        "a_post_1": [0, 1, 0, 0, 1, 1, 0, 1],
                        "b_post_1": [0, 1, 0, 0, 1, 1, 0, 1],
                    },
                    index=self.taxa,
                ),
                pd.DataFrame(
                    {
                        "a_post_2": [0, 0, 1, 0, 1, 1, 0, 0],
                        "b_post_2": [0, 0, 1, 0, 1, 1, 0, 0],
                    },
                    index=self.taxa,
                ),
                pd.DataFrame(
                    {
                        "a_post_3": [0, 1, 0, 1, 1, 1, 1, 0],
                        "b_post_3": [0, 1, 0, 1, 1, 1, 1, 0],
                    },
                    index=self.taxa,
                ),
            ],
            "tree": (
                "((lost:1,other_baseline:1):1,"
                "(transient:1,(colonizer:1,early_and_colonizer:1):1):1)root;"
            ),
        }

    def test_run_phylogenetic_test(self):
        outputs = run_phylogenetic_test(self.dataset)

        self.assertEqual(set(outputs["results"]), {"subject_a", "subject_b"})
        self.assertEqual(
            outputs["results"]["subject_a"]["categories"],
            {
                "lost_species": {"lost"},
                "other_baseline": {"other_baseline"},
                "transient": {"transient"},
                "colonizers": {"colonizer", "early_and_colonizer"},
            },
        )
        self.assertListEqual(
            list(outputs["unifrac_similarities"].columns),
            [
                "colonizers_lost",
                "colonizers_other_baseline",
                "transient_lost",
                "transient_other_baseline",
                "lost_species_other_baseline",
                "colonizers_transient",
            ],
        )
        self.assertEqual(outputs["unifrac_similarities"].shape, (2, 6))


class TestFunctionalTest(TestCase):
    """This class tests the FunctionalTest class."""
    def setUp(self) -> None:
        self.path = "C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_results/picrust2_out_pipeline/EC_metagenome_out/pred_metagenome_contrib.tsv.gz"
        self.df = pd.read_csv(self.path, sep="\t",
                              compression="gzip" if self.path.endswith(".gz") else None)
        self.metadata = pd.read_csv('C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_results/Metadata.csv')
        self.name_to_run = self.metadata.set_index("Name")["Run"].to_dict()
        self.sample_name = self.name_to_run['77_Day37_CZA']
        self.chunksize = 100
        self.functional_test_path = FunctionalTest(self.path, self.sample_name, self.chunksize)
        self.functional_test_df = FunctionalTest(self.df, self.sample_name, self.chunksize)
        self.taxa = ['0f1670c0893a582f95f3b17d38e371f2']
        self.species_table_path = self.functional_test_path.get_sample_species_table(self.taxa)
        self.species_table_df = self.functional_test_df.get_sample_species_table(self.taxa)

    def test_init(self):
        self.assertTrue(isinstance(self.functional_test_path.reader, TextFileReader))
        self.assertTrue(isinstance(self.functional_test_df.sample_table, pd.DataFrame))

    def test_get_sample_species_table(self):
        self.assertListEqual(self.taxa, list(set(self.species_table_df["taxon"].tolist())))
        self.assertListEqual(self.taxa, list(set(self.species_table_path["taxon"].tolist())))

class TestApplyFunctionalTest(TestCase):
    """This class tests the ApplyFunctionalTest class."""
    def setUp(self) -> None:
        self.path = "C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_results/picrust2_out_pipeline_strat/EC_metagenome_out/pred_metagenome_contrib.tsv.gz"
        self.metadata = pd.read_csv('C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_results/Metadata.csv')
        self.name_to_run = self.metadata.set_index("Name")["Run"].to_dict()
        self.base_col = "77_Day1_CZA"
        self.abx_col = "77_Day6_CZA"
        self.post_cols = ["77_Day9_CZA", "77_Day12_CZA", "77_Day16_CZA", "77_Day25_CZA", "77_Day37_CZA"]
        self.unstratified_path = "C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_results/picrust2_out_pipeline_strat/EC_metagenome_out/pred_metagenome_unstrat.tsv.gz"

        self.fun_unstrat = pd.read_csv(self.unstratified_path, sep="\t",
                                       compression="gzip" if self.unstratified_path.endswith(".gz") else None,
                                       index_col=0)
        self.data = pd.read_csv('C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_results/feature_table_rarified.csv',
                                index_col=0)
        self.sur_cols = ['135_Day37_PTZD1', '33_Day37_Ct', '132_Day1_CtD2']
        self.iters = 2
        self.verbose = True
        self.n_jobs = -1
        self.apply_functional_test_simple = ApplyFunctionalTest(self.fun_unstrat, Path(self.path), self.name_to_run,
                                                                self.base_col, self.abx_col, self.post_cols,
                                                                self.sur_cols, self.data, self.iters, self.verbose,
                                                                self.n_jobs, new_type="simple")
        self.apply_functional_test_soft = ApplyFunctionalTest(self.fun_unstrat, Path(self.path), self.name_to_run,
                                                              self.base_col, self.abx_col, self.post_cols,
                                                              self.sur_cols, self.data, self.iters, self.verbose,
                                                              self.n_jobs, new_type="soft")
        self.apply_functional_test_strict = ApplyFunctionalTest(self.fun_unstrat, Path(self.path), self.name_to_run,
                                                                self.base_col, self.abx_col, self.post_cols,
                                                                self.sur_cols, self.data, self.iters, self.verbose,
                                                                self.n_jobs, new_type="strict")
    def _find_new_null_iters(self):
        self.assertTrue(all(v == len(self.apply_functional_test_simple.new) for v in [
            len(n) for n in self.apply_functional_test_simple.new_null_cont]))
        self.assertTrue(all(v == len(self.apply_functional_test_soft.new) for v in [
            len(n) for n in self.apply_functional_test_soft.new_null_cont]))
        self.assertTrue(all(v == len(self.apply_functional_test_strict.new) for v in [
            len(n) for n in self.apply_functional_test_strict.new_null_cont]))
        self.assertTrue(all(set(v).isdisjoint(self.apply_functional_test_simple.dis
                                              ) for v in self.apply_functional_test_simple.fun_new_null_cont))
        self.assertTrue(all(set(v).isdisjoint(self.apply_functional_test_soft.dis
                                              ) for v in self.apply_functional_test_soft.fun_new_null_cont))
        self.assertTrue(all(set(v).isdisjoint(self.apply_functional_test_strict.dis
                                              ) for v in self.apply_functional_test_strict.fun_new_null_cont))

    def test_pad_fun_grouped(self):
        self.assertEqual(np.size(self.apply_functional_test_simple.fun_new_grouped_pad),
                         np.size(self.apply_functional_test_simple.fun_dis_grouped_pad))

    def test_normalize_pad_fun_grouped(self):
        self.assertEqual(np.sum(self.apply_functional_test_simple.fun_new_grouped_pad_norm), 1)
        self.assertEqual(np.sum(self.apply_functional_test_simple.fun_dis_grouped_pad_norm), 1)

class TestUnifracTest(TestCase):
    """
    This class tests the UnifracTest class.
    """
    def setUp(self) -> None:
        self.feature_table = pd.DataFrame({
            'S1_D1_ABX_A': [1, 0, 1, 1, 0, 1],
            'S2_D1_ABX_B': [1, 1, 1, 1, 1, 0],
            'S1_D6_ABX_A': [1, 0, 0, 1, 0, 0],
            'S2_D6_ABX_B': [0, 0, 0, 0, 1, 1],
            'S1_D180_ABX_A': [1, 1, 0, 1, 0, 1],
            'S2_D180_ABX_B': [1, 1, 1, 1, 0, 1]
        })

        otu_ids = ['OTU1', 'OTU2', 'OTU3', 'OTU4', 'OTU5', 'OTU6']

        self.feature_table.index = otu_ids

        def build_mock_tree(otu_ids):

            otu_ids = [otu for otu in otu_ids if otu != 'OTU6']

            leaves = [TreeNode(name=otu) for otu in otu_ids]

            root = TreeNode(name="root")
            for leaf in leaves:
                leaf.length = 1.0
                root.append(leaf)

            return root

        self.tree = build_mock_tree(otu_ids)

        print(self.tree)

        self.test_ids = ['S1', 'S2']

        self.pattern = "_D"

        self.days = ["1", "6", "180"]

        self.add_info = ["ABX_A", "ABX_B", "ABX_A", "ABX_B", "ABX_A", "ABX_B"]

        self.unifrac_object = UnifracTest(self.feature_table, self.tree, self.test_ids, self.pattern, self.days,
                                          self.add_info)

        self.new = self.unifrac_object._find_new('S1_D1_ABX_A', 'S1_D6_ABX_A', 'S1_D180_ABX_A')
        self.dis = self.unifrac_object._find_dis('S1_D1_ABX_A', 'S1_D6_ABX_A', 'S1_D180_ABX_A')

    def test_filter_biom(self):
        self.assertListEqual(self.unifrac_object.shared_otus, ['OTU1', 'OTU2', 'OTU3', 'OTU4', 'OTU5'])
        self.assertListEqual(list(self.unifrac_object.filtered_biom.ids(axis='sample')), ['S1_D1_ABX_A',
                                                                                               'S2_D1_ABX_B',
                                                                                               'S1_D6_ABX_A',
                                                                                               'S2_D6_ABX_B',
                                                                                               'S1_D180_ABX_A',
                                                                                               'S2_D180_ABX_B'])
    def test_find_new(self):
        self.assertTrue(np.array_equal(self.new, np.array([False, True, False, False, False])))

    def test_find_dis(self):
        self.assertTrue(np.array_equal(self.dis, np.array([False, False, True, False, False])))

    def test_find_new_null(self):
        new_null = self.unifrac_object._find_new_null(self.new, self.dis, 'S2_D180_ABX_B')
        self.assertEqual(np.sum(new_null), 1)
        self.assertTrue(new_null[2] is not True)

class TestSimilarityCorrelation(TestCase):
    def setUp(self) -> None:
        keys = ["A", "B", "C", "D", "E"]
        abx = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
        base = np.array([[1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                         [1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                         [0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                         [1, 1, 1, 0, 1, 0, 0, 1, 1, 1]])
        self.post_ABX_container = {"A": np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                                                  [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                                                  [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]]),
                                   "B": np.array([[1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                                                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                                                  [1, 1, 1, 1, 1, 1, 0, 0, 1, 1]]),
                                   "C": np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                                                  [0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                                                  [0, 1, 0, 1, 1, 1, 0, 1, 1, 1]]),
                                   "D": np.array([[1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                                                  [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                                                  [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]]),
                                   "E": np.array([[1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                                                  [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
                                                  [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]])
                                   }
        keys_ref = ["A", "B", "C", "D", "E", "F", "G"]
        base_others = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
        base_ref = np.vstack([base, base_others])
        self.baseline_ref = pd.DataFrame(base_ref.T, columns=keys_ref)
        self.ABX = pd.DataFrame(abx.T, columns=keys)
        self.baseline = pd.DataFrame(base.T, columns=keys)
        self.method = "Jaccard"
        self.timepoints = 1
        self.iters = 10
        self.new = False
        self.sim = SimilarityCorrelation(self.ABX, self.baseline, self.post_ABX_container, self.baseline_ref,
                                         self.method, self.timepoints, self.iters, self.new, True)
        self.sim_soft = SimilarityCorrelation(self.ABX, self.baseline, self.post_ABX_container, self.baseline_ref,
                                              self.method, self.timepoints, self.iters, self.new, False)
        self.timepoints_vals = self.sim._returned_species(self.sim.baseline[0, :], self.sim.ABX[0, :],
                                                          self.sim.post_ABX_matrices[0])
        self.timepoints_vals_soft = self.sim_soft._returned_species(self.sim.baseline[1, :], self.sim.ABX[1, :],
                                                                    self.sim.post_ABX_matrices[1])

    def test_generate_synthetic_cohorts(self):
        cohorts_lst = self.sim.synthetic_baseline_lst
        for cohort in cohorts_lst:
            # check if the constraints are satisfied
            self.assertTrue(np.array_equal(cohort.sum(axis=1), self.baseline_ref.to_numpy().T.sum(axis=1)))

    def test_returned_species(self):
        # strict
        self.assertEqual(len(self.timepoints_vals), self.post_ABX_container["A"].shape[0])
        self.assertTrue(np.array_equal(self.timepoints_vals[0], np.array([False, False, False, False, False, False,
                                                                          True, False, False, False])))
        self.assertTrue(np.array_equal(self.timepoints_vals[1], np.array([False, False, False, False, True, False,
                                                                          False, False, True, False])))
        self.assertTrue(np.array_equal(self.timepoints_vals[2], np.array([False, False, False, False, False, False,
                                                                          False, False, False, False])))
        # soft
        self.assertEqual(len(self.timepoints_vals_soft), self.post_ABX_container["B"].shape[0])
        self.assertTrue(np.array_equal(self.timepoints_vals_soft[0], np.array([False, False, True, False, False, False,
                                                                               False, False, True, False])))
        self.assertTrue(np.array_equal(self.timepoints_vals_soft[1], np.array([False, True, False, False, False, False,
                                                                               False, False, False, True])))
        self.assertTrue(np.array_equal(self.timepoints_vals_soft[2], np.array([False, False, False, False, False, True,
                                                                               False, False, False, False])))

    def test_find_subset(self):
        # check if the number of timepoints is correct
        idx = self.sim._find_subset(self.sim.baseline[0, :], self.sim.ABX[0, :], self.sim.post_ABX_matrices[0],
                                    self.timepoints_vals)
        self.assertEqual(np.size(idx), self.post_ABX_container["A"].shape[0] - self.timepoints)


class Test_NullModel(TestCase):
    def setUp(self) -> None:
        self.baseline_sample = np.array([0.1, 0, 0, 0.2, 0.3, 0.1, 0, 0.1, 0.2, 0])
        self.ABX_sample = np.array([0, 0, 0.3, 0.1, 0, 0.1, 0, 0, 0, 0.5])
        self.test_matrix = np.array([[0.5, 0, 0.1, 0, 0.2, 0.2, 0, 0, 0, 0],
                                    [0.5, 0, 0.1, 0, 0.2, 0.1, 0, 0.1, 0, 0]])
        self.baseline = np.array([[0.1, 0, 0, 0.2, 0.3, 0.1, 0, 0.1, 0.2, 0],
                                  [0.5, 0.1, 0, 0, 0.1, 0.1, 0, 0.1, 0.1, 0],
                                  [0.1, 0, 0.2, 0.1, 0, 0.1, 0, 0, 0.2, 0.3]])
        self.num_reals = 3
        self.timepoints = 1
        self.null_model = NullModel(self.baseline_sample, self.ABX_sample, self.baseline,
                                    self.test_matrix, self.num_reals, self.timepoints)

    def test_find_subset(self):
        self.assertListEqual(list(self.null_model.subset_comp), [0, 2, 4, 5])
        self.assertListEqual(list(self.null_model.subset), [1, 3, 6, 7, 8, 9])
        self.assertEqual(np.size(self.null_model.subset_comp) + np.size(self.null_model.subset), self.baseline.shape[1])

    def test_distance(self):
        # Test the constraint
        self.assertEqual(np.size(np.nonzero(self.null_model.synthetic_samples[0, self.null_model.subset])),
                         np.size(np.nonzero(self.test_matrix[-1, :][self.null_model.subset])))
        self.assertEqual(self.null_model.distance(method="Specificity")[0], 0)


class Test_Surrogate(TestCase):
    def setUp(self) -> None:
        self.test_post_abx_matrix = np.array([[1, 0, 1, 0, 0, 0, 0, 1],
                                              [1, 0, 1, 1, 0, 1, 0, 1],
                                              [1, 0, 0, 0, 0, 0, 1, 1],
                                              [1, 1, 1, 1, 1, 1, 1, 0]])
        self.test_base_samples_collection = {"Test subject": np.array([[1, 0, 1, 1, 1, 0, 1, 1]])}
        self.test_base_samples_collection_mat = {"Test subject": np.array([[1, 0, 1, 1, 1, 0, 1, 1],
                                                                           [1, 0, 1, 0, 1, 0, 1, 1],
                                                                           [1, 0, 1, 1, 1, 1, 0, 1]])}
        self.base_samples_collections = {"Subject A": np.array([[0, 1, 1, 1, 0, 1, 1, 0]]),
                                         "Subject B": np.array([[1, 1, 0, 1, 1, 0, 0, 1]])}
        self.base_samples_collections_mat = {"Subject A": np.array([[0, 1, 1, 1, 0, 1, 1, 0],
                                                                    [0, 1, 1, 1, 0, 1, 0, 0]]),
                                             "Subject B": np.array([[1, 1, 0, 1, 1, 0, 0, 1],
                                                                    [1, 1, 0, 1, 1, 0, 0, 0]])}
        self.test_abx_sample = np.array([[1, 1, 0, 0, 0, 0, 0, 1]])
        self.timepoints = 1
        self.surrogate = Surrogate(self.base_samples_collections, self.test_base_samples_collection,
                                   self.test_post_abx_matrix, self.test_abx_sample, timepoints=self.timepoints)
        self.results = self.surrogate.apply_surrogate_data_analysis()
        self.surrogate_not_strict = Surrogate(self.base_samples_collections, self.test_base_samples_collection,
                                          self.test_post_abx_matrix, self.test_abx_sample, timepoints=self.timepoints,
                                          strict=False)
        self.results_not_strict = self.surrogate_not_strict.apply_surrogate_data_analysis()
        self.surrogate_null = Surrogate(self.base_samples_collections, self.test_base_samples_collection,
                                        self.test_post_abx_matrix, self.test_abx_sample, timepoints=0)
        self.results_null = self.surrogate_null.apply_surrogate_data_analysis()
        self.surrogate_null_not_strict = Surrogate(self.base_samples_collections, self.test_base_samples_collection,
                                                   self.test_post_abx_matrix, self.test_abx_sample, timepoints=0,
                                                   strict=False)
        self.results_null_not_strict = self.surrogate_null_not_strict.apply_surrogate_data_analysis()
        self.surrogate_mat = Surrogate(self.base_samples_collections_mat, self.test_base_samples_collection_mat,
                                       self.test_post_abx_matrix, self.test_abx_sample, timepoints=self.timepoints)
        self.results_mat = self.surrogate_mat.apply_surrogate_data_analysis()

    def test_find_subset(self):
        self.assertListEqual(list(self.surrogate.subset), [2, 3, 4, 5, 6, 7])
        self.assertListEqual(list(self.surrogate_not_strict.subset), [3, 4, 5, 6, 7])
        self.assertListEqual(list(self.surrogate_null.subset), [2, 3, 4, 5, 6, 7])
        self.assertListEqual(list(self.surrogate_null_not_strict.subset), [2, 3, 4, 5, 6, 7])

    def test_matrix_case(self):
        self.assertEqual(self.surrogate_mat.test_key, "Test subject")
        self.assertListEqual(list(self.surrogate_mat.test_base_sample), [1, 0, 1, 1, 1, 0, 1, 1])

