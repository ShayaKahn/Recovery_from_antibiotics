import pandas as pd
import numpy as np

class Rarify:
    """
    This class rarifies an OTU table to a desired sequencing depth.
    """
    def __init__(self, df, depth=None):
        """
        Inputs:
        df: dataframe, represents OTU table with samples as columns and OTUs as rows,
            the values are the counts (integers)
        depth: int, the number of reads to be sampled from each sample, if None,
               the minimum number of reads is the default. The number of reads to be sampled from each sample
        """
        self.df, self.depth = self._validate_input(df, depth)
        self.sampling_dict = self._construct_sampling_dict()

    def _validate_input(self, df, depth):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The input object is not a DataFrame.")
        if not df.applymap(lambda x: isinstance(x, int)).all().all():
            raise ValueError("Not all values in the DataFrame are integers.")
        if depth is not None:
            if not isinstance(depth, int):
                raise TypeError("Depth must be an integer.")
            if not (df.sum(axis=0).min() < depth < df.sum(axis=0).max()):
                raise ValueError(f"Depth must be between {df.sum(axis=0).min()} and {df.sum(axis=0).max()}.")
        OTU_table = df
        if depth is None:
            # define the depth as the minimum number of reads in the samples
            OTU_depth = OTU_table.sum(axis=0).min()
        else:
            # filter out samples with lower number of reads then the depth
            OTU_depth = depth
            OTU_table = self._filter_low_frequancy(OTU_table, OTU_depth)
        return OTU_table, OTU_depth

    @staticmethod
    def _filter_low_frequancy(OTU_table, depth):
        """
        Filter out samples with lower frequancy then the depth.
        Inputs:
        As described in the __init__ method.
        Return:
        df_filtered: dataframe, represents OTU table with samples as columns and OTUs as rows,
                     the values are the counts (integers)
        """
        # filter out samples with lower frequancy then the depth
        df_filtered = OTU_table.loc[:, (OTU_table.sum(axis=0) >= depth)]
        return df_filtered

    def _construct_sampling_dict(self):
        """
        Construct a sampling dictionary.
        Return:
        sampling_dict: dictionary, keys are the sample names and values are the indices of the OTUs in the OTU table
        repeated the number of times they are present in the sample.
        """
        # construct a sampling dictionary
        sampling_dict = {
            col: np.repeat(self.df.index, self.df[col]).tolist()
            for col in self.df.columns
        }
        return sampling_dict

    def rarify(self):
        """
        Rarify the OTU table to the desired depth.
        Return:
        rar_df: dataframe, represents the rarified OTU table with samples as columns and OTUs as rows,
                the values are the counts (integers)
        """
        # rarify the dataframe
        rar_df = pd.DataFrame(0, index=self.df.index, columns=self.df.columns)

        for col in self.df.columns:
            selected_indices = np.random.choice(self.sampling_dict[col], self.depth, replace=False)
            np.add.at(rar_df[col].values, selected_indices, 1)
        return rar_df