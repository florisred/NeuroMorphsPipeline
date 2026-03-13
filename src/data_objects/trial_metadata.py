import pandas as pd
import numpy as np
import numpy.typing as npt


class TrialMetadata:
    def __init__(self, metadata_df: pd.DataFrame = None):
        if metadata_df is None:
            self.metadata_df = pd.DataFrame()
        else: self.metadata_df = metadata_df
        self.trial_lens = []
        self._masked_metadata = None
        self._use_mask = False

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
        df['src_cat'] = np.where(
            df['stim_type'] == 'anchor',
            np.nan,
            df['src_cat']
        )
        df['dst_cat'] = np.where(
            df['stim_type'] == 'anchor',
            np.nan,
            df['dst_cat']
        )
        df['step_index'] = np.where(
            df['stim_type'] == 'anchor',
            np.nan,
            df['src_cat']
        )
        df['norm_step'] = np.where(
            df['stim_type'] == 'anchor',
            np.nan,
            df['src_cat']
        )
        df.index = df['morph_name']
        df.rename_axis('morph', inplace=True)
        self.trial_lens.append(df.shape[0])
        self.metadata_df = pd.concat([self.metadata_df, df])


    def synchronize_with_data(self, combined_df: pd.DataFrame):
        """
        Drops all the trials that are not in the provided dataframe. Useful when having combined multiple sessions
        where not all shown trials were in each session.
        """
        metadata_lookup = self.metadata_df.drop_duplicates(subset='morph_name')

        self.metadata_df = metadata_lookup.reindex(combined_df.index).rename_axis(index='morph')

    def get_morph_names(self, as_list:bool = False) -> pd.Series|list:
        """
        Returns the morph names of the loaded metadata in pd.Series format
        """
        if as_list: return self.get_metadata()['morph_name'].values
        return self.get_metadata()['morph_name']

    def get_pair_keys(
            self,
            unique: bool = True,
            dropna: bool = True,
            values: bool = False
        ) -> npt.NDArray[np.str_] | pd.Series:
        """
        Returns the pair_keys of the loaded metadata
        """

        pair_keys = self.metadata_df['pair_key']
        if dropna:
            pair_keys = pair_keys.dropna()
        if unique:
            pair_keys = pair_keys.unique()
        if values:
            return pair_keys
        else:
            return pair_keys


    def get_shared_morphs(self):
        """
        returns all morphs that are in all sessions
        """
        morphs_per_session = []
        prev_session_index = 0
        for i, session_index in enumerate(self.trial_lens):
            morphs_in_session = self.get_morph_names()[prev_session_index:prev_session_index + session_index]
            unique_morphs = set(morphs_in_session.unique())
            morphs_per_session.append(unique_morphs)
            prev_session_index += session_index
        shared_morphs = set.intersection(*morphs_per_session)
        return list(shared_morphs)

    def apply_mask(self, mask):
        self._masked_metadata = self.metadata_df[mask]
        self._use_mask = True

    def disable_mask(self):
        self._use_mask = False

    def get_anchor_mask(self):
        mask = self.get_metadata()['stim_type'] == 'anchor'
        return mask

    def get_anchors(self):
        return self.get_metadata()[self.get_anchor_mask()]

    def get_anchor_names(self):
        return self.get_anchors()['morph_name']

    def get_metadata(self):
        if self._use_mask: return self._masked_metadata
        else: return self.metadata_df

    def sort(self, sorted_idx):
        """
        sorts either the masked data or real data, based on if the mask is used.
        """
        if self._use_mask:
            if len(self._masked_metadata) != len(sorted_idx):
                raise ValueError("the length of the sorted idx does not match the length of the metadata")
            self._masked_metadata = self._masked_metadata.iloc[sorted_idx]
        else:
            if len(self.metadata_df) != len(sorted_idx):
                raise ValueError("the length of the sorted idx does not match the length of the metadata")
            self.metadata_df = self.metadata_df.iloc[sorted_idx]

    def find_matching_pair_keys(self, search_term: str) -> tuple[list, int]:
        """
        Finds the indexes where the pair key matches the search_term
        :param search_term: str: the pair key to look for
        :return: tuple[list, int]: the indexes where the pair key matches the search_term, and the number of matches
        """
        pair_keys = self.get_pair_keys(unique=False, dropna=False, values=True)
        matches = [i for i, pair_key in enumerate(pair_keys) if search_term in str(pair_key)]
        return matches, len(matches)


