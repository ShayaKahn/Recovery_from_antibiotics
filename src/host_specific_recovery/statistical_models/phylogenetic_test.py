from __future__ import annotations
from io import StringIO
from pathlib import Path
from typing import List, Set, Union
import numpy as np
import pandas as pd
from skbio import TreeNode
from skbio.diversity import beta_diversity


class PhylogeneticTest:
    """This class is responsible to implement the phylogenetic test for a given subject."""

    __slots__ = ("base", "abx", "post", "tree", "timepoints_post")

    def __init__(
            self,
            base_cols: Union[pd.Series, pd.DataFrame],
            abx_col: Union[pd.Series, pd.DataFrame],
            post_cols: pd.DataFrame,
            tree: Union[str, Path, TreeNode, None] = None,
    ) -> None:
        self.base = self._as_sample_table(base_cols, "base_cols")
        self.abx = self._as_single_sample_table(abx_col, "abx_col")
        self.post = post_cols
        self.tree = self._parse_tree(tree)
        self.timepoints_post = post_cols.shape[1]

    @staticmethod
    def _parse_tree(tree: Union[str, Path, TreeNode, None]) -> Union[TreeNode, None]:
        if tree is None or isinstance(tree, TreeNode):
            return tree

        if isinstance(tree, Path):
            return TreeNode.read(str(tree), format="newick", convert_underscores=False)

        tree_path = Path(tree)
        try:
            is_tree_path = tree_path.exists()
        except OSError:
            is_tree_path = False

        if is_tree_path:
            return TreeNode.read(str(tree_path), format="newick", convert_underscores=False)

        return TreeNode.read(StringIO(tree), format="newick", convert_underscores=False)

    @staticmethod
    def _as_sample_table(samples: Union[pd.Series, pd.DataFrame], name: str) -> pd.DataFrame:
        if isinstance(samples, pd.Series):
            return samples.to_frame()
        if isinstance(samples, pd.DataFrame):
            return samples
        raise TypeError(f"{name} must be a pandas Series or DataFrame.")

    @classmethod
    def _as_single_sample_table(cls, sample: Union[pd.Series, pd.DataFrame], name: str) -> pd.DataFrame:
        sample_table = cls._as_sample_table(sample, name)
        if sample_table.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one sample.")
        return sample_table

    def _presence_tables(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        taxa = self.base.index.union(self.abx.index).union(self.post.index)

        base = self.base.reindex(taxa, fill_value=0).gt(0)
        abx = self.abx.reindex(taxa, fill_value=0).gt(0)
        post = self.post.reindex(taxa, fill_value=0).gt(0)

        return base, abx, post

    def find_lost_taxa(self) -> List[str]:
        return sorted(self.find_lost_species())

    def find_lost_species(self) -> Set[str]:
        base, abx, post = self._presence_tables()

        mask = (base.all(axis=1)
                & ~abx.any(axis=1)
                & ~post.any(axis=1))

        return set(mask.index[mask])

    def find_other_baseline(self) -> Set[str]:
        base, _, _ = self._presence_tables()
        lost_species = self.find_lost_species()

        mask = base.all(axis=1)

        return set(mask.index[mask]) - lost_species

    def find_transient(self) -> Set[str]:
        base, abx, post = self._presence_tables()

        if post.shape[1] < 2:
            return set()

        absent_before_post = ~base.any(axis=1) & ~abx.any(axis=1)
        present_once_in_post = post.sum(axis=1).eq(1)
        absent_in_last_post_sample = ~post.iloc[:, -1]

        mask = absent_before_post & present_once_in_post & absent_in_last_post_sample

        return set(mask.index[mask])

    def find_colonizers(self) -> Set[str]:
        base, abx, post = self._presence_tables()

        if post.shape[1] < 2:
            return set()

        absent_before_post = ~base.any(axis=1) & ~abx.any(axis=1)
        present_in_last_two_post_samples = post.iloc[:, -2:].all(axis=1)

        mask = absent_before_post & present_in_last_two_post_samples

        return set(mask.index[mask])

    def find_species_categories(self) -> dict[str, Set[str]]:
        return {
            "lost_species": self.find_lost_species(),
            "other_baseline": self.find_other_baseline(),
            "transient": self.find_transient(),
            "colonizers": self.find_colonizers(),
        }

    def _unifrac_similarity_between_taxa_sets(self, first: Set[str], second: Set[str]) -> float:
        if self.tree is None:
            raise ValueError("A phylogenetic tree is required to compute UniFrac similarity.")

        if not first or not second:
            return np.nan

        tree_taxa = {tip.name for tip in self.tree.tips()}
        taxa = sorted((first | second) & tree_taxa)

        if not taxa:
            raise ValueError("None of the selected taxa are present in the phylogenetic tree.")

        first_filtered = first & set(taxa)
        second_filtered = second & set(taxa)
        if not first_filtered or not second_filtered:
            return np.nan

        counts = pd.DataFrame(
            [
                [int(taxon in first_filtered) for taxon in taxa],
                [int(taxon in second_filtered) for taxon in taxa],
            ],
            index=["first", "second"],
            columns=taxa,
        )

        distance = beta_diversity(
            metric="unweighted_unifrac",
            counts=counts,
            ids=counts.index,
            tree=self.tree,
            taxa=taxa,
            validate=True,
        ).data[0, 1]

        return 1.0 - distance

    def compute_unifrac_similarities(self) -> dict[str, float]:
        categories = self.find_species_categories()

        return {
            "colonizers_lost": self._unifrac_similarity_between_taxa_sets(
                categories["colonizers"], categories["lost_species"]
            ),
            "colonizers_other_baseline": self._unifrac_similarity_between_taxa_sets(
                categories["colonizers"], categories["other_baseline"]
            ),
            "transient_lost": self._unifrac_similarity_between_taxa_sets(
                categories["transient"], categories["lost_species"]
            ),
            "transient_other_baseline": self._unifrac_similarity_between_taxa_sets(
                categories["transient"], categories["other_baseline"]
            ),
            "lost_species_other_baseline": self._unifrac_similarity_between_taxa_sets(
                categories["lost_species"], categories["other_baseline"]
            ),
            "colonizers_transient": self._unifrac_similarity_between_taxa_sets(
                categories["colonizers"], categories["transient"]
            ),
        }
