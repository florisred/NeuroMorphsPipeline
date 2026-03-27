from pathlib import Path
from data_objects.trial_metadata import TrialMetadata
from data_objects.data_source import DataSource
from MixIns.two_photon_mixin import TwoPhotonMixIn
import pandas as pd
from utils.utils import scale_session, split_morphs
import logging

logger = logging.getLogger(__name__)


class TwoPhotonDataSource(DataSource, TwoPhotonMixIn):

    def __init__(self, file_paths: list[Path], data_location: str):
        super().__init__(file_paths)
        self.data_location = data_location
        self.labels_list = ['pair_key', 'step_index', 'src_cat', 'dst_cat']
        self.data_type = 'TwoPhoton'


    def load_data(self, split:bool=False, seed:int=42):
        """
        Loads the .h5 data from each session and loads them in the metadata file.
        Then, it groups the data from trials with identical stimuli together and averages the mean activation
        """
        data_dfs = []
        self._split = split
        for session_dir in self.file_paths:
            raw_data_df, raw_meta_df = self._load_h5_file(session_dir, self.data_location, self.labels_list)
            temp_meta = TrialMetadata()
            temp_meta.process_and_append(raw_meta_df)
            raw_morph_names = temp_meta.get_morph_names()
            temp_meta_df = temp_meta.metadata_df
            raw_data_df.index = raw_morph_names
            if split: raw_data_df, temp_meta_df = split_morphs(raw_trials=raw_data_df, raw_metadata=temp_meta_df, seed=seed)
            data_dfs.append(raw_data_df)
            self.metadata.append(temp_meta_df)


        processed_dfs = []
        raw_data_dfs = []
        shared_morphs = self.metadata.get_shared_morphs()
        for data_df in data_dfs:
            filtered_df = data_df[data_df.index.isin(shared_morphs)]
            scaled_session = scale_session(filtered_df)
            session_mean = scaled_session.groupby('morph_name').mean()
            processed_dfs.append(session_mean)
            sorted_scaled_session = scaled_session.sort_index()
            raw_data_dfs.append(sorted_scaled_session)
        self._raw_data = raw_data_dfs
        combined_df = pd.concat(processed_dfs, axis=1, join='inner')
        self.metadata.synchronize_with_data(combined_df)
        self._data = combined_df
        logger.info(f"Loaded {len(combined_df)} morphs with {combined_df.shape[1]} neurons")






