import pandas as pd
import numpy as np
import numpy.typing as npt


class TrialMetadata:
    def __init__(self):
        self.metadata_df = pd.DataFrame()

    def process_and_append(self, raw_trials_metadata_df: pd.DataFrame):
        """
        adds metadata from each session, and only keeps the ones already in the list if there is already a metadata_df
        a raw_trials_metadata_df should have the following format:
        each individual trial should be a row
        the following columns should exist:
            - pair_key (alphabetically left to right)
            - step_index (int)
            - src_cat
            - dst_cat
        """
        df = raw_trials_metadata_df.copy()
        df['step_index'] = df['step_index'].astype(int)
        min_step_index = min(df['step_index'].values)
        max_step_index = max(df['step_index'].values)
        df['norm_step'] = round(df['step_index'] / (max_step_index - min_step_index), 2)
        is_src = df['step_index'] == min_step_index
        is_dst = df['step_index'] == max_step_index
        df['stim_type'] = np.where(
            is_src | is_dst,
            'anchor',
            'morph'
        )
        partial_names = (
                df['src_cat'] + "_" +
                (1-df['norm_step'].astype(float)).round(2).astype(str) + "_" +
                df['dst_cat'] + "_" +
                df['norm_step'].astype(float).round(2).astype(str)
        )
        df['morph_name'] = np.where(
            is_src,
            df['src_cat'],
            np.where(is_dst, df['dst_cat'], partial_names)
        )
        df.loc[df['stim_type'] == 'anchor', 'pair_key'] = np.nan
        self.metadata_df = pd.concat([self.metadata_df, df], ignore_index=True)

    def synchronize_with_data(self, combined_df: pd.DataFrame):
        """
        Drops all the trials that are not in the provided dataframe. Useful when having combined multiple sessions
        where not all shown trials were in each session.
        """
        self.metadata_df = self.metadata_df.drop_duplicates(subset='morph_name')
        index_df = combined_df.index.to_frame(index=False, name='morph_name')
        self.metadata_df = index_df.merge(self.metadata_df, on='morph_name', how='left')

    def get_morph_names(self) -> pd.Series:
        """
        Returns the morph names of the loaded metadata in pd.Series format
        """
        return self.metadata_df['morph_name']

    def get_pair_keys(
            self,
            unique: bool = True,
            dropna: bool = True,
            as_series: bool = False
        ) -> npt.NDArray[np.str_] | pd.Series:
        """
        Returns the pair_keys of the loaded metadata
        """

        pair_keys = self.metadata_df['pair_key']
        if dropna:
            pair_keys = pair_keys.dropna()
        if unique:
            pair_keys = pair_keys.unique()
        if as_series:
            return pd.Series(pair_keys)
        else:
            return pair_keys



