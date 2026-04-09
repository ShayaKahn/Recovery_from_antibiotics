import pandas as pd
import numpy as np
from typing import Tuple, List, Union, Sequence, Optional
from collections import defaultdict
from pathlib import Path

import duckdb

class FunctionalDataPipeline:
    __slots__ = ("fun_by_sample_table", "rename_map", "stratified_fun", "ASV_table", "con")

    def __init__(self, fun_by_sample_table: pd.DataFrame, stratified_fun: Union[Path, str],
                 rename_map: Union[dict, None] = None,
                 ASV_table: Optional[pd.DataFrame] = None) -> None:
        """
        This class works on picrust2 functional output data. It assumes that the function-by-sample table is the
        unstratified table, and stratified_fun is the stratified table. The ASV_table is
        optional and used for taxon filtering. This table should be the table used to generate the functional profiles.

        :param fun_by_sample_table: DataFrame with functions as rows and samples as columns.
        :param rename_map: {Run_id -> sample_name}, optional. Relevant if sample names in the functional inputs are Run_ids.
        :param stratified_fun: Path to stratified function table (tsv/tsv.gz) with columns:
            sample, function, taxon, taxon_abun, taxon_function_abun, genome_function_count.
        :param ASV_table: DataFrame with ASVs as rows and samples as columns.
        """
        # validate inputs
        self._validate_inputs(fun_by_sample_table, stratified_fun, rename_map, ASV_table)

        self.rename_map = rename_map

        # rename samples in function-by-sample table
        if self.rename_map is None:
            self.fun_by_sample_table = fun_by_sample_table
        else:
            self.fun_by_sample_table = fun_by_sample_table.rename(columns=rename_map).astype(float)

        # rename samples in ASV table if provided
        if ASV_table is not None:
            if self.rename_map is not None:
                self.ASV_table = ASV_table.rename(columns=rename_map).astype(float)
            else:
                self.ASV_table = ASV_table

        # validate inputs
        if isinstance(stratified_fun, str):
            self.stratified_fun = Path(stratified_fun)
        else:
            self.stratified_fun = stratified_fun

        self.con = duckdb.connect(":memory:")
        if self.rename_map is not None:
            self._map_sample_names_strat()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "FunctionalDataPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _validate_inputs(fun_by_sample_table, stratified_fun, rename_map, ASV_table) -> None:
        # validate fun_by_sample_table
        if not isinstance(fun_by_sample_table, pd.DataFrame):
            raise TypeError("fun_by_sample_table must be a pandas DataFrame")
        # validate stratified_fun
        if not isinstance(stratified_fun, (Path, str)):
            raise TypeError("stratified_fun must be a pathlib.Path or string")
        # validate rename_map
        if rename_map is not None and not isinstance(rename_map, dict):
            raise TypeError("rename_map must be a dict or None")
        # validate ASV_table
        if ASV_table is not None and not isinstance(ASV_table, pd.DataFrame):
            raise TypeError("ASV_table must be a pandas DataFrame or None")

    def _map_sample_names_strat(self) -> None:
        """
        Create a new stratified file where column 'sample' is renamed using self.rename_map,
        then set self.stratified_fun to that new path.
        """
        temp_file = self.stratified_fun.parent / f"temp_{self.stratified_fun.name}"

        # build rename_map table inside DuckDB
        rm = pd.DataFrame({"run": list(self.rename_map.keys()), "name": list(self.rename_map.values())})
        self.con.register("rename_map_df", rm)
        self.con.execute("CREATE OR REPLACE TEMP TABLE rename_map AS SELECT * FROM rename_map_df")
        self.con.unregister("rename_map_df")

        self.con.execute(
            """
            COPY (
                SELECT
                  COALESCE(rm.name, s.sample)::VARCHAR AS sample,
                  s.function::VARCHAR AS function,
                  s.taxon::VARCHAR AS taxon,
                  s.taxon_abun::FLOAT AS taxon_abun,
                  s.taxon_function_abun::FLOAT AS taxon_function_abun,
                  s.genome_function_count::FLOAT AS genome_function_count
                FROM read_csv_auto(?, delim='\t', header=true) s
                LEFT JOIN rename_map rm
                  ON s.sample = rm.run
            )
            TO ? (DELIMITER '\t', HEADER true, COMPRESSION gzip)
            """,
            [str(self.stratified_fun), str(temp_file)],
        )

        self.stratified_fun = temp_file

    def subsample_stratified_fun(self, sample_name: Union[str, Sequence[str]]) -> pd.DataFrame:
        """
        Extract all rows for one sample (str) or multiple samples (Sequence[str]) from the stratified function table.

        Returns a pandas DataFrame with:
        sample, function, taxon, taxon_abun, taxon_function_abun, genome_function_count
        """

        if isinstance(sample_name, str):
            df = self.con.execute(
                """
                SELECT sample, function, taxon, taxon_abun, taxon_function_abun, genome_function_count
                FROM read_csv_auto(?, delim='\t', header=true)
                WHERE sample = ?
                """,
                [str(self.stratified_fun), sample_name],
            ).df()

        elif isinstance(sample_name, Sequence):
            names = list(sample_name)
            if not names:
                raise ValueError("sample_name sequence is empty")

            df = self.con.execute(
                """
                SELECT sample, function, taxon, taxon_abun, taxon_function_abun, genome_function_count
                FROM read_csv_auto(?, delim='\t', header=true)
                WHERE sample IN (SELECT * FROM UNNEST(?))
                """,
                [str(self.stratified_fun), names],
            ).df()

        else:
            raise TypeError("sample_name must be a str or a sequence of str")

        if df.empty:
            raise ValueError(f"Sample '{sample_name}' not found in file: {self.stratified_fun}")

        df["sample"] = df["sample"].astype("string")
        df["function"] = df["function"].astype("string")
        df["taxon"] = df["taxon"].astype("string")
        for c in ["taxon_abun", "taxon_function_abun", "genome_function_count"]:
            df[c] = df[c].astype("float32")

        return df

    def filter_taxon_sample_rows(self, sample_rows: pd.DataFrame, selected_taxa: List,
                                 keep: bool = True) -> pd.DataFrame:
        """
        Filter sample_rows DataFrame to keep or remove selected_taxa.

        :param sample_rows: DataFrame with rows for a given sample.
        :param selected_taxa: List of taxa to keep or remove.
        :param keep: If True, keep selected_taxa; if False, remove selected_taxa.
        :return: Filtered and aggregated DataFrame.
        """
        if keep:
            sample_rows_filtered = sample_rows.loc[sample_rows["taxon"].isin(selected_taxa)].copy()
        else:
            sample_rows_filtered = sample_rows.loc[~sample_rows["taxon"].isin(selected_taxa)].copy()

        sample = sample_rows_filtered["sample"].iat[0]

        sample_rows_filtered_agg = (sample_rows_filtered.groupby("function", as_index=True)[
            "taxon_function_abun"].sum().to_frame(name=sample))

        return sample_rows_filtered_agg

    def filter_taxon_sample_rows_cond(self, sample_rows: pd.DataFrame, taxa_distribution: List,
                                      n: int) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[None, None]]:
        """
        Filter sample_rows DataFrame to keep N random taxa from taxa_distribution with non-zero abundance.

        :param sample_rows: DataFrame with rows for a given sample.
        :param taxa_distribution: List of taxa to select from which to choose.
        :param n: Number of taxa to select.
        :return: Tuple of filtered and aggregated DataFrame and Series of selected taxa abundances.
        """

        # isolate taxa with non-zero abundance
        nonzero_taxa = set(sample_rows.loc[sample_rows["taxon_abun"] > 0, "taxon"].unique())
        eligible_taxa = nonzero_taxa.intersection(taxa_distribution)

        if len(eligible_taxa) < n:
            return None, None

        # select top N random taxa from eligible_taxa
        rng = np.random.default_rng()
        selected_taxa = list(rng.choice(list(eligible_taxa), size=n, replace=False))

        sample_rows_filtered_agg, selected_taxa_abun = self.filter_taxon_sample_rows(sample_rows, selected_taxa,
                                                                                     keep=True)

        return sample_rows_filtered_agg, selected_taxa_abun

    def taxa_function_dict(self, sample_rows: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Create a dictionary mapping each taxon to a Series of its function counts.

        :param sample_rows: DataFrame with rows for a given sample.
        :return: Dictionary mapping taxon to Series of function counts.
        """

        dup = sample_rows.duplicated(["taxon", "function"], keep=False)
        if dup.any():
            bad = sample_rows.loc[dup, ["taxon", "function"]].drop_duplicates().head(10)
            raise ValueError(f"Found duplicate (taxon, function) pairs. Examples:\n{bad}")

        taxa_fun_dict = {}
        for taxon, sub in sample_rows.groupby("taxon", sort=False):
            s = sub.set_index("function")["genome_function_count"]
            s = pd.to_numeric(s, errors="coerce")
            taxa_fun_dict[str(taxon)] = s
        return taxa_fun_dict

    def taxa_dict_to_matrix(self, taxa_fun_dict: dict[str, pd.Series], fill_value: float = 0.0) -> pd.DataFrame:
        """
        Convert a dictionary of taxon function Series to a DataFrame matrix.

        :param taxa_fun_dict: Dictionary mapping taxon to Series of function counts.
        :param fill_value: Value to fill missing entries in the matrix.
        :return: DataFrame matrix with functions as rows and taxa as columns.
        """

        mat = pd.concat(taxa_fun_dict, axis=1)
        mat = mat.apply(pd.to_numeric, errors="coerce").fillna(fill_value)

        return mat

    def find_baseline_taxa(self, base_cols: List[str]) -> set:
        """
        Find taxa that are present in all base_cols.

        :param base_cols: baseline columns of a specific subject
        :return: Set of baseline taxa
        """
        if self.ASV_table is None:
            print("ASV_table is None. Skipping find_new_taxa.")
            return None

        mask = self.ASV_table[base_cols].gt(0).all(axis=1)

        return set(self.ASV_table.index[mask])

    def find_new_taxa(self, base_cols: List[str], abx_cols: List[str], post_col: str) -> Union[set, None]:
        """
        Find taxa that are new in post_col compared to base_cols and abx_col.

        :param base_cols: baseline columns of a specific subject
        :param abx_cols: antibiotic treatment columns of a specific subject
        :param post_col: post-antibiotic treatment column of a specific subject
        :return: Set of new taxa
        """
        if self.ASV_table is None:
            print("ASV_table is None. Skipping find_new_taxa.")
            return None

        mask = (self.ASV_table[base_cols].eq(0).all(axis=1)
                & self.ASV_table[abx_cols].eq(0).all(axis=1)
                & (self.ASV_table[post_col] > 0))

        return set(self.ASV_table.index[mask])

    def find_lost_taxa(self, base_cols: List[str], abx_cols: List[str], post_cols: List[str]) -> Union[set, None]:
        """
        Find taxa that are lost in post_cols compared to base_cols and abx_col.

        :param base_cols: baseline columns of a specific subject
        :param abx_cols: antibiotic treatment columns of a specific subject
        :param post_cols: post-antibiotic treatment columns of a specific subject
        :return: Set of lost taxa
        """
        if self.ASV_table is None:
            print("ASV_table is None. Skipping find_new_taxa.")
            return None

        mask = (self.ASV_table[base_cols].gt(0).all(axis=1)
                & (self.ASV_table[abx_cols[-1]] == 0)
                & (self.ASV_table[post_cols].eq(0).all(axis=1)))

        return set(self.ASV_table.index[mask])

    def find_colonizers(self, base_cols: List[str], abx_cols: List[str], post_cols: List[str],
                        k: int = 2) -> Tuple[set, set]:
        """
        Find colonizer and transient taxa based on their presence in post_cols.

        :param base_cols: baseline columns of a specific subject
        :param abx_cols: antibiotic treatment columns of a specific subject
        :param post_cols: post-antibiotic treatment columns of a specific subject
        :param k: Number of last post_cols to consider for transient detection.
        :return: Tuple of sets: (colonizers, transient)
        """
        # common mask for colonizers and transient taxa: absence in baseline and abx
        standard_mask = (self.ASV_table.loc[:, base_cols].eq(0).all(axis=1) &
                         self.ASV_table.loc[:, abx_cols].eq(0).all(axis=1))

        # Define the traget for the transient taxa
        target_transient = self.ASV_table.loc[:, post_cols[-k]]
        others_transient = self.ASV_table.loc[:, post_cols[:-k] + post_cols[-k + 1:]]

        transient_mask = (standard_mask &
                          target_transient.gt(0) &
                          others_transient.eq(0).all(axis=1))

        # Define the target for the colonizer taxa
        colonizers_mask = (self.ASV_table.loc[:, post_cols[-2]] > 0) & (self.ASV_table.loc[:, post_cols[-1]] > 0)

        # Extract taxa sets
        colonizers = set(self.ASV_table.index[colonizers_mask])
        transient = set(self.ASV_table.index[transient_mask])

        return colonizers, transient

    def build_taxa_series(self) -> dict[str, pd.Series]:
        """
        Build dict: taxon -> pd.Series(index=function, values=genome_function_count)
        from the current self.stratified_fun using DuckDB (in-memory connection self.con).
        """
        query = (
            "SELECT taxon::VARCHAR AS taxon, function::VARCHAR AS function, "
            "ANY_VALUE(genome_function_count)::FLOAT AS genome_function_count "
            "FROM read_csv_auto(?, delim='\t', header=true) "
            "GROUP BY taxon, function"
        )

        rows = self.con.execute(query, [str(self.stratified_fun)]).fetchall()

        taxa_fun_dict = defaultdict(list)
        for taxon, fun, cnt in rows:
            taxa_fun_dict[taxon].append((fun, float(cnt)))

        taxa_series_dict = {
            taxon: pd.Series(
                data=[cnt for _, cnt in pairs],
                index=[fun for fun, _ in pairs],
                dtype="float32",
                name=taxon,
            )
            for taxon, pairs in taxa_fun_dict.items()
        }

        return taxa_series_dict
