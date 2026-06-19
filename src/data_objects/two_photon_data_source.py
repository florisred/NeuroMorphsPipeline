from pathlib import Path
from data_objects.trial_metadata import TrialMetadata
from data_objects.data_source import DataSource
import pandas as pd
from utils.utils import scale_session, split_morphs, load_h5_file
import logging
import numpy as np


logger = logging.getLogger(__name__)


class TwoPhotonDataSource(DataSource):

    def __init__(self, file_paths: list[Path], data_location: str, metadata_locations:list[str]|None = None):
        super().__init__(file_paths)
        self._data_location = data_location
        if metadata_locations is None:
            self._labels_list = ['y/pair_key', 'y/step_index', 'y/src_cat', 'y/dst_cat']
        else:
            self._labels_list = metadata_locations
        self._data_type = 'TwoPhoton'

    def load_data(self, seed: int = 42, train_percent = 0.7):
        """
        Loads the .h5 data from each session and loads them in the metadata file.
        Then, it isolates shared morphs across sessions and interpolates repetitions
        so that every session matches the maximum repetition count for a given morph.
        """
        data_dfs = []
        for session_dir in self.file_paths:
            raw_data_df, raw_meta_df = load_h5_file(session_dir, self._data_location, self._labels_list)
            temp_meta = TrialMetadata()
            temp_meta.process_and_append(raw_meta_df)
            raw_morph_names = temp_meta.morph_names
            temp_meta_df = temp_meta._metadata_df
            raw_data_df.index = raw_morph_names
            data_dfs.append(raw_data_df)
            self._metadata.append(temp_meta_df)
        shared_morphs = self.metadata.shared_morphs
        processed_sessions = []
        for data_df in data_dfs:
            filtered_df = data_df[data_df.index.isin(shared_morphs)]
            scaled_session = scale_session(filtered_df)
            processed_sessions.append(scaled_session)
        max_reps_per_morph = {}
        for p_df in processed_sessions:
            rep_counts = p_df.index.value_counts()
            for morph, count in rep_counts.items():
                max_reps_per_morph[morph] = max(max_reps_per_morph.get(morph, 0), count)


        def interpolate_trials(morph_df, target_reps):
            n_current = len(morph_df)
            if n_current == target_reps:
                return morph_df.reset_index(drop=True)
            if n_current == 1:
                return pd.concat([morph_df] * target_reps).reset_index(drop=True)
            old_idx = np.linspace(0, 1, n_current)
            new_idx = np.linspace(0, 1, target_reps)
            temp_df = morph_df.copy()
            temp_df.index = old_idx
            combined_idx = np.union1d(old_idx, new_idx)
            interp_df = temp_df.reindex(combined_idx).interpolate(method='index')
            result_df = interp_df.loc[new_idx].copy()
            result_df.index = range(target_reps)
            return result_df
        aligned_data_dfs = []
        for p_df in processed_sessions:
            session_morph_dfs = []
            for morph in sorted(list(shared_morphs)):
                morph_df = p_df.loc[[morph]]
                target_reps = max_reps_per_morph[morph]
                interp_df = interpolate_trials(morph_df, target_reps)
                interp_df['morph_name'] = morph
                session_morph_dfs.append(interp_df)
            session_aligned = pd.concat(session_morph_dfs, ignore_index=True)
            aligned_data_dfs.append(session_aligned)
        for i in range(len(aligned_data_dfs)):
            if i == 0: continue
            aligned_data_dfs[i] = aligned_data_dfs[i].drop(columns='morph_name')
        combined_df = pd.concat(aligned_data_dfs, axis=1)
        combined_df.set_index(keys='morph_name', inplace=True)
        self._metadata.synchronize_with_data(combined_df)
        self._data = combined_df
        logger.info(f"Loaded {len(combined_df)} morph trials with {combined_df.shape[1]} neurons")


