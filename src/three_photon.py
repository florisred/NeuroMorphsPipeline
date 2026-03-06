from itertools import groupby
from os.path import join
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import calc_mean_per_stimulus, scale_session
from sklearn.decomposition import PCA
import itertools


class Twophoton:
    def __init__(self):
        self.data_df = None
        self.labels = None
        self.processed_df = None
        self.labels_list = ["y/pair_key", "y/step_index", "y/src_cat", "y/dst_cat", "y/norm_step", "y/stim_type"]

    def load_2p_data(self, session_dirs, data_location):
        """

        """
        data_dfs = []
        for session_dir in session_dirs:

            data, labels = self._load_h5_file(
                session_dir=session_dir,
                data_location=data_location
            )
            grouped_data, index_df = calc_mean_per_stimulus(data, labels)
            data_dfs.append(grouped_data)

        combined_df, self.labels = self._concatenate(
            data_dfs=data_dfs,
            labels=labels
        )
        self.data_df = scale_session(combined_df)

        data_dict = {
            'data': self.data_df,
            'labels': self.labels
        }
        return data_dict

    def _load_h5_file(self, session_dir, data_location):
        two_photon_folder = join(session_dir, "2p_data")
        file = list(Path(two_photon_folder).glob("*.h5"))
        if len(file) != 1: raise AssertionError(f"Need exactly one .hy file in {two_photon_folder}")
        f = h5py.File(file[0], 'r')
        data = pd.DataFrame(f[data_location]).dropna(axis=0, how='all')
        labels = self._load_metadata(f)
        return data, labels


    def _load_metadata(self, f):
        metadata_dataframe = pd.DataFrame()
        for metadata_location in self.labels_list:
            meta_name = metadata_location.split("/")[-1]
            metadata_array = np.array(f[metadata_location]).flatten()
            metadata_dataframe[meta_name] = metadata_array.astype(str)
        metadata_dataframe = metadata_dataframe.T
        metadata_df = metadata_dataframe.T
        is_src = metadata_df['stim_type'] == 'endpoint_src'
        is_dst = metadata_df['stim_type'] == 'endpoint_dst'
        partial_names = (
                metadata_df['src_cat'] + "_" +
                (1-metadata_df['norm_step'].astype(float)).round(2).astype(str) + "_" +
                metadata_df['dst_cat'] + "_" +
                metadata_df['norm_step'].astype(float).round(2).astype(str)
        )
        metadata_df['morph_name'] = np.where(
            is_src,
            metadata_df['src_cat'],
            np.where(is_dst, metadata_df['dst_cat'], partial_names)
        )
        metadata_df['stim_type'] = np.where(
            is_src | is_dst,
            'anchor',
            'morph'
        )
        metadata_df.loc[metadata_df['stim_type'] == 'anchor', 'pair_key'] = np.nan
        return metadata_df

    @staticmethod
    def _concatenate(data_dfs, labels):
        combined_df = pd.concat(data_dfs, axis=1, join='inner')
        index_df= combined_df.index.to_frame(index=False, name='morph_name')
        labels_df = index_df.merge(labels.drop_duplicates('morph_name'), on='morph_name', how='left')
        return combined_df, labels_df



