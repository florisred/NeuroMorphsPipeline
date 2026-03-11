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

    def load_2p_data(self, session_dirs, data_location: str):
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




