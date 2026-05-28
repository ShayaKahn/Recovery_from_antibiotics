from src.host_specific_recovery.statistical_models.phylogenetic_test import PhylogeneticTest
from io import StringIO
from typing import Any
import pandas as pd
from Bio import Phylo


def _tree_to_newick(tree: Any) -> Any:
    if tree is None or isinstance(tree, str):
        return tree

    handle = StringIO()
    Phylo.write(tree, handle, "newick")
    return handle.getvalue()


def _subject_names(dataset: dict, n_subjects: int) -> list[str]:
    names = dataset.get("filtered_keys", dataset.get("keys"))
    if names is None:
        return [f"Subject {i + 1}" for i in range(n_subjects)]
    return [str(name) for name in list(names)[:n_subjects]]


def run_phylogenetic_test(dataset: dict) -> dict:
    baseline_frame = dataset["baseline_df"]
    abx_frame = dataset["abx_df"]
    post_abx_frames = dataset["post_abx_cohorts_df"]
    tree = _tree_to_newick(dataset.get("tree"))

    n_subjects = min(
        baseline_frame.shape[1],
        abx_frame.shape[1],
        *(frame.shape[1] for frame in post_abx_frames),
    )
    subject_names = _subject_names(dataset, n_subjects)

    results = {}
    objects = {}
    rows = []

    for i, subject_name in enumerate(subject_names):
        base_cols = baseline_frame.iloc[:, i]
        abx_col = abx_frame.iloc[:, i]
        post_cols = pd.concat(
            [frame.iloc[:, i] for frame in post_abx_frames],
            axis=1,
        )

        phylogenetic_test = PhylogeneticTest(base_cols, abx_col, post_cols, tree)
        categories = phylogenetic_test.find_species_categories()

        similarities = phylogenetic_test.compute_unifrac_similarities()

        results[subject_name] = {
            "categories": categories,
            "unifrac_similarities": similarities,
        }
        objects[subject_name] = phylogenetic_test
        rows.append({"subject": subject_name, **similarities})

    return {
        "results": results,
        "objects": objects,
        "unifrac_similarities": pd.DataFrame(rows).set_index("subject"),
    }
