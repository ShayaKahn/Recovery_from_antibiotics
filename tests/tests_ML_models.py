from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut

from src.host_specific_recovery.ML_models.linear_regression import MicrobiomeRegularizedLinearRegressor
from src.host_specific_recovery.utils.microbiome_transformers import (
    PrevalenceFilter,
    combine_taxonomic_functional_tables,
)


class TestMicrobiomeRegularizedLinearRegressor(TestCase):
    @staticmethod
    def _standardized_target(observed, other_values):
        other_values = np.asarray(other_values, dtype=float)
        return (observed - other_values.mean()) / other_values.std(ddof=0)

    @staticmethod
    def _column_positions(row_full_index, allowed_full_indices):
        return [
            full_index if full_index < row_full_index else full_index - 1
            for full_index in allowed_full_indices
            if full_index != row_full_index
        ]

    def test_leave_one_out_targets_exclude_only_held_out_model_subject(self):
        X = pd.DataFrame(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            columns=["taxonomic__otu_1", "taxonomic__otu_2"],
        )

        sample_similarity_indices = np.array([0, 2, 4])
        similarity_mid = np.array([10.0, 20.0, 30.0])

        # Full baseline subjects are [0, 1, 2, 3, 4].
        # Rows are aligned with X/filtered_keys, while each row omits its own
        # full subject index from the columns:
        # row 0, full subject 0 -> columns [1, 2, 3, 4]
        # row 1, full subject 2 -> columns [0, 1, 3, 4]
        # row 2, full subject 4 -> columns [0, 1, 2, 3]
        similarity_others_mid = np.array([
            [101.0, 102.0, 103.0, 104.0],
            [200.0, 201.0, 203.0, 204.0],
            [300.0, 301.0, 302.0, 303.0],
        ])

        model = MicrobiomeRegularizedLinearRegressor(
            min_prevalence=0,
            model_type="ridge",
            n_jobs=1,
            avoid_target_leakage=True,
            similarity_mid=similarity_mid,
            similarity_others_mid=similarity_others_mid,
            sample_similarity_indices=sample_similarity_indices,
        )

        _, predictions = model.evaluate_cv_predictions(
            X,
            None,
            cv=LeaveOneOut(),
            return_predictions=True,
        )

        expected_y_true = []
        full_indices = np.arange(similarity_others_mid.shape[1] + 1)
        for row_index, row_full_index in enumerate(sample_similarity_indices):
            allowed_full_indices = full_indices[full_indices != row_full_index]
            columns = self._column_positions(row_full_index, allowed_full_indices)
            expected_y_true.append(
                self._standardized_target(
                    similarity_mid[row_index],
                    similarity_others_mid[row_index, columns],
                )
            )

        self.assertTrue(np.allclose(predictions["y_true"].to_numpy(), expected_y_true))


class TestCombineTaxonomicFunctionalTables(TestCase):
    def test_combine_prefixes_columns_and_preserves_taxonomic_sample_order(self):
        taxonomic_df = pd.DataFrame(
            {"otu_1": [1.0, 2.0, 3.0], "otu_2": [4.0, 5.0, 6.0]},
            index=["s1", "s2", "s3"],
        )
        # functional_df is provided in a different row order than taxonomic_df.
        functional_df = pd.DataFrame(
            {"path_1": [30.0, 10.0, 20.0]},
            index=["s3", "s1", "s2"],
        )

        combined = combine_taxonomic_functional_tables(taxonomic_df, functional_df)

        self.assertListEqual(list(combined.index), ["s1", "s2", "s3"])
        self.assertListEqual(
            list(combined.columns), ["taxonomic__otu_1", "taxonomic__otu_2", "functional__path_1"]
        )
        self.assertListEqual(combined["functional__path_1"].tolist(), [10.0, 20.0, 30.0])

    def test_combine_adds_metadata_and_preserves_taxonomic_sample_order(self):
        taxonomic_df = pd.DataFrame({"otu_1": [1.0, 2.0]}, index=["s1", "s2"])
        functional_df = pd.DataFrame({"path_1": [10.0, 20.0]}, index=["s1", "s2"])
        metadata_df = pd.DataFrame(
            {"age": [42, 31], "sex": ["female", "male"]},
            index=["s2", "s1"],
        )

        combined = combine_taxonomic_functional_tables(
            taxonomic_df, functional_df, metadata_df=metadata_df
        )

        self.assertListEqual(list(combined.index), ["s1", "s2"])
        self.assertListEqual(
            list(combined.columns),
            ["taxonomic__otu_1", "functional__path_1", "metadata__age", "metadata__sex"],
        )
        self.assertListEqual(combined["metadata__age"].tolist(), [31, 42])
        self.assertListEqual(combined["metadata__sex"].tolist(), ["male", "female"])

    def test_combine_raises_on_sample_ids_missing_from_functional_table(self):
        taxonomic_df = pd.DataFrame({"otu_1": [1.0, 2.0]}, index=["s1", "s2"])
        functional_df = pd.DataFrame({"path_1": [10.0]}, index=["s1"])

        with self.assertRaises(ValueError):
            combine_taxonomic_functional_tables(taxonomic_df, functional_df)

    def test_combine_raises_on_sample_ids_missing_from_taxonomic_table(self):
        taxonomic_df = pd.DataFrame({"otu_1": [1.0]}, index=["s1"])
        functional_df = pd.DataFrame({"path_1": [10.0, 20.0]}, index=["s1", "s2"])

        with self.assertRaises(ValueError):
            combine_taxonomic_functional_tables(taxonomic_df, functional_df)

    def test_combine_raises_on_duplicated_sample_ids(self):
        taxonomic_df = pd.DataFrame({"otu_1": [1.0, 2.0]}, index=["s1", "s1"])
        functional_df = pd.DataFrame({"path_1": [10.0, 20.0]}, index=["s1", "s2"])

        with self.assertRaises(ValueError):
            combine_taxonomic_functional_tables(taxonomic_df, functional_df)

    def test_combine_raises_on_metadata_sample_id_mismatch(self):
        taxonomic_df = pd.DataFrame({"otu_1": [1.0, 2.0]}, index=["s1", "s2"])
        functional_df = pd.DataFrame({"path_1": [10.0, 20.0]}, index=["s1", "s2"])
        metadata_df = pd.DataFrame({"age": [31, 42]}, index=["s1", "s3"])

        with self.assertRaises(ValueError):
            combine_taxonomic_functional_tables(taxonomic_df, functional_df, metadata_df)


class TestMicrobiomeRegularizedLinearRegressorCombinedInput(TestCase):
    @staticmethod
    def _build_tables(n_samples, seed=0):
        rng = np.random.default_rng(seed)
        index = [f"s{i}" for i in range(n_samples)]
        taxonomic_df = pd.DataFrame(
            rng.random((n_samples, 3)), columns=["otu_1", "otu_2", "otu_3"], index=index
        )
        functional_df = pd.DataFrame(
            rng.random((n_samples, 2)), columns=["path_1", "path_2"], index=index
        )
        y = rng.random(n_samples)
        return taxonomic_df, functional_df, y

    def test_fit_and_predict_work_with_combined_dataframe(self):
        taxonomic_df, functional_df, y = self._build_tables(n_samples=6)
        X = combine_taxonomic_functional_tables(taxonomic_df, functional_df)

        model = MicrobiomeRegularizedLinearRegressor(min_prevalence=0.0, model_type="ridge", n_jobs=1)
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions.shape[0], X.shape[0])
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_fit_and_predict_support_numeric_and_categorical_metadata(self):
        taxonomic_df, functional_df, y = self._build_tables(n_samples=6)
        metadata_df = pd.DataFrame(
            {
                "age": [30.0, 40.0, np.nan, 35.0, 50.0, 45.0],
                "sex": ["female", "male", "female", None, "male", "female"],
            },
            index=taxonomic_df.index,
        )
        X = combine_taxonomic_functional_tables(taxonomic_df, functional_df, metadata_df)

        model = MicrobiomeRegularizedLinearRegressor(
            min_prevalence=0.0, model_type="ridge", n_jobs=1
        )
        model.fit(X, y)
        predictions = model.predict(X)
        importance = model.feature_importance()

        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertIn("metadata", set(importance["feature_type"]))
        metadata_features = set(
            importance.loc[importance["feature_type"] == "metadata", "feature"]
        )
        self.assertIn("age", metadata_features)
        self.assertTrue(any(name.startswith("sex_") for name in metadata_features))

    def test_metadata_encoder_ignores_unseen_categories_at_prediction_time(self):
        taxonomic_df, functional_df, y = self._build_tables(n_samples=6)
        metadata_df = pd.DataFrame(
            {"group": ["a", "b", "a", "b", "a", "b"]},
            index=taxonomic_df.index,
        )
        X_train = combine_taxonomic_functional_tables(
            taxonomic_df, functional_df, metadata_df
        )

        model = MicrobiomeRegularizedLinearRegressor(
            min_prevalence=0.0, model_type="ridge", n_jobs=1
        )
        model.fit(X_train, y)

        X_new = X_train.iloc[[0]].copy()
        X_new.loc[:, "metadata__group"] = "unseen"
        prediction = model.predict(X_new)

        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction[0]))

    def test_metadata_binary_features_are_not_scaled(self):
        taxonomic_df, functional_df, y = self._build_tables(n_samples=6)
        metadata_df = pd.DataFrame(
            {
                "binary": [0, 1, 0, 1, 0, 1],
                "continuous": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            },
            index=taxonomic_df.index,
        )
        X = combine_taxonomic_functional_tables(taxonomic_df, functional_df, metadata_df)

        model = MicrobiomeRegularizedLinearRegressor(
            min_prevalence=0.0, model_type="ridge", n_jobs=1
        )
        model.fit(X, y)

        metadata_transformer = (
            model.pipeline_.named_steps["preprocessor"].named_transformers_["metadata"]
        )
        transformed = metadata_transformer.transform(X[["metadata__binary", "metadata__continuous"]])
        transformed_names = metadata_transformer.get_feature_names_out()
        binary_position = list(transformed_names).index("binary__metadata__binary")
        continuous_position = list(transformed_names).index("continuous__metadata__continuous")

        np.testing.assert_array_equal(transformed[:, binary_position], metadata_df["binary"])
        self.assertAlmostEqual(float(np.mean(transformed[:, continuous_position])), 0.0)
        self.assertAlmostEqual(float(np.std(transformed[:, continuous_position])), 1.0)

    def test_fit_and_predict_work_with_functional_only_dataframe(self):
        _, functional_df, y = self._build_tables(n_samples=6)
        X = functional_df.add_prefix("functional__")

        model = MicrobiomeRegularizedLinearRegressor(min_prevalence=0.0, model_type="ridge", n_jobs=1)
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions.shape[0], X.shape[0])
        self.assertTrue(np.all(np.isfinite(predictions)))

        importance = model.feature_importance()
        self.assertSetEqual(set(importance["feature_type"]), {"functional"})
        self.assertSetEqual(set(importance["feature"]), set(functional_df.columns))

    def test_predict_output_order_follows_input_row_order(self):
        taxonomic_df, functional_df, y = self._build_tables(n_samples=6)
        X = combine_taxonomic_functional_tables(taxonomic_df, functional_df)

        model = MicrobiomeRegularizedLinearRegressor(min_prevalence=0.0, model_type="ridge", n_jobs=1)
        model.fit(X, y)

        reference_predictions = model.predict(X)
        permuted_predictions = model.predict(X.loc[X.index[::-1]])

        np.testing.assert_allclose(permuted_predictions, reference_predictions[::-1])

    def test_preprocessing_is_fitted_only_on_training_folds(self):
        taxonomic_df, functional_df, y = self._build_tables(n_samples=6)
        X = combine_taxonomic_functional_tables(taxonomic_df, functional_df)

        model = MicrobiomeRegularizedLinearRegressor(min_prevalence=0.0, model_type="ridge", n_jobs=1)
        cv = KFold(n_splits=3, shuffle=False)

        seen_fit_sizes = []
        original_fit = PrevalenceFilter.fit

        def recording_fit(self_filter, X_in, y_in=None):
            seen_fit_sizes.append(len(X_in))
            return original_fit(self_filter, X_in, y_in)

        with patch.object(PrevalenceFilter, "fit", recording_fit):
            model.evaluate_cv_predictions(X, y, cv=cv)

        train_fold_size = X.shape[0] - X.shape[0] // cv.get_n_splits()
        self.assertTrue(seen_fit_sizes)
        self.assertTrue(all(size == train_fold_size for size in seen_fit_sizes))
        self.assertNotIn(X.shape[0], seen_fit_sizes)

    def test_feature_importance_retains_taxonomic_and_functional_feature_types(self):
        taxonomic_df, functional_df, y = self._build_tables(n_samples=6)
        X = combine_taxonomic_functional_tables(taxonomic_df, functional_df)

        model = MicrobiomeRegularizedLinearRegressor(min_prevalence=0.0, model_type="ridge", n_jobs=1)
        model.fit(X, y)
        importance = model.feature_importance()

        self.assertSetEqual(set(importance["feature_type"]), {"taxonomic", "functional"})
        self.assertTrue(
            all(not name.startswith("taxonomic__") and not name.startswith("functional__")
                for name in importance["feature"])
        )
        self.assertSetEqual(
            set(importance.loc[importance["feature_type"] == "taxonomic", "feature"]),
            set(taxonomic_df.columns),
        )
        self.assertSetEqual(
            set(importance.loc[importance["feature_type"] == "functional", "feature"]),
            set(functional_df.columns),
        )

    def test_predict_on_new_samples_does_not_need_original_training_tables(self):
        taxonomic_df, functional_df, y = self._build_tables(n_samples=6, seed=0)
        X_train = combine_taxonomic_functional_tables(taxonomic_df, functional_df)

        model = MicrobiomeRegularizedLinearRegressor(min_prevalence=0.0, model_type="ridge", n_jobs=1)
        model.fit(X_train, y)

        new_taxonomic_df, new_functional_df, _ = self._build_tables(n_samples=2, seed=1)
        new_taxonomic_df.index = ["new_1", "new_2"]
        new_functional_df.index = ["new_1", "new_2"]
        X_new = combine_taxonomic_functional_tables(new_taxonomic_df, new_functional_df)

        # only the combined table is available at prediction time; the per-branch
        # tables used to build it are discarded to prove predict() does not need them.
        del new_taxonomic_df, new_functional_df

        predictions = model.predict(X_new)

        self.assertEqual(predictions.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(predictions)))
